"""
Segmentation Module — SegFormer (Cityscapes, B5 backbone)

Maximum-accuracy path detection using:
  - Multi-scale inference (test-time augmentation)
  - Horizontal flip augmentation
  - Sliding window for high-res inputs
  - Guided-filter edge refinement
  - Morphological post-processing

Cityscapes class IDs we care about:
  0  = road
  1  = sidewalk
  9  = terrain (grass verge)
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from typing import Tuple, List, Optional


# ---------------------------------------------------------------------------
# Cityscapes class definitions
# ---------------------------------------------------------------------------

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

# Classes that are traversable (safe to travel on)
TRAVERSABLE_CLASS_IDS = {
    0,   # road
    1,   # sidewalk
    9,   # terrain (grass verge — include but lower priority)
}

# Classes that are definite obstacles (used to clip the traversable mask)
OBSTACLE_CLASS_IDS = {
    2,   # building
    3,   # wall
    4,   # fence
    11,  # person
    12,  # rider
    13,  # car
    14,  # truck
    15,  # bus
    16,  # train
    17,  # motorcycle
    18,  # bicycle
}

# Color map for visualization (BGR)
CITYSCAPES_COLORS_BGR = {
    0:  (232, 35,   244),   # road       — will be overridden to green
    1:  (160, 170,  96),    # sidewalk
    2:  (70,  70,   70),    # building
    8:  (107, 142,  35),    # vegetation
    9:  (152, 251, 152),    # terrain
    10: (70,  130, 180),    # sky
    11: (220,  20,  60),    # person
    13: (0,    0,  142),    # car
    6:  (81,    0,  81),    # ground
}


class SegmentationModel:

    def __init__(self, config: dict, device: torch.device):
        self.config  = config
        self.device  = device
        self.class_names = CITYSCAPES_CLASSES
        self.processor   = None
        self.model       = None

        # Multi-scale / augmentation settings
        self.multi_scale   = config.get("multi_scale", True)
        self.scales        = config.get("scales", [0.75, 1.0, 1.25])
        self.flip_augment  = config.get("flip_augment", True)
        self.use_sliding   = config.get("use_sliding_window", False)
        self.win_size      = config.get("window_size", 1024)
        self.win_stride    = config.get("window_stride", 768)
        self.use_fp16      = config.get("use_fp16", False)

        # Post-processing settings
        self.close_k       = config.get("morphology_close_k", 17)
        self.open_k        = config.get("morphology_open_k", 9)
        self.top_n         = config.get("keep_top_n_components", 3)
        self.edge_refine   = config.get("edge_refinement", True)

        self._load()

    # ------------------------------------------------------------------
    # Model loading — SegFormer only
    # ------------------------------------------------------------------

    def _load(self):
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        model_name = self.config.get(
            "segformer_model",
            "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
        )
        print(f"[Seg] Loading SegFormer: {model_name}")
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device).eval()

        if self.use_fp16 and self.device.type == "cuda":
            self.model = self.model.half()
            print("[Seg] Using FP16 precision")

        # Log VRAM usage after load
        if self.device.type == "cuda":
            mem_mb = torch.cuda.memory_allocated() / 1e6
            print(f"[Seg] GPU memory allocated: {mem_mb:.0f} MB")

        print("[Seg] SegFormer loaded ✓")

    # ------------------------------------------------------------------
    # Inference — multi-scale + flip TTA
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Args:
            rgb_image: HxWx3 uint8 RGB
        Returns:
            seg_map: HxW int64 class ids (Cityscapes 19-class)
        """
        H, W = rgb_image.shape[:2]

        if self.multi_scale:
            return self._predict_multiscale(rgb_image, H, W)
        elif self.use_sliding:
            return self._predict_sliding_window(rgb_image, H, W)
        else:
            return self._predict_single(rgb_image, H, W)

    def _predict_single(self, rgb: np.ndarray, H: int, W: int) -> np.ndarray:
        """Single-scale inference."""
        from PIL import Image
        pil_img = Image.fromarray(rgb)
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)

        if self.use_fp16 and self.device.type == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].half()

        outputs = self.model(**inputs)
        logits = outputs.logits  # (1, num_classes, H/4, W/4)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        seg = logits.argmax(dim=1).squeeze(0)
        return seg.cpu().numpy().astype(np.int64)

    def _get_logits(self, rgb: np.ndarray, H: int, W: int) -> torch.Tensor:
        """Get raw logits (before argmax) for a single image, resized to (H, W)."""
        from PIL import Image
        pil_img = Image.fromarray(rgb)
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)

        if self.use_fp16 and self.device.type == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].half()

        outputs = self.model(**inputs)
        logits = outputs.logits  # (1, C, H/4, W/4)
        logits = F.interpolate(
            logits.float(), size=(H, W), mode="bilinear", align_corners=False
        )
        return logits  # (1, C, H, W)

    def _predict_multiscale(self, rgb: np.ndarray, H: int, W: int) -> np.ndarray:
        """
        Multi-scale test-time augmentation:
        1. Run inference at each scale
        2. Optionally run on horizontally flipped input
        3. Average the logits across all runs
        4. Take argmax for final prediction

        This significantly improves boundary accuracy at the cost of
        ~N× inference time (N = len(scales) × (2 if flip else 1)).
        """
        accumulated_logits = None
        count = 0

        for scale in self.scales:
            # Scale the image
            sH, sW = int(H * scale), int(W * scale)
            if sH < 32 or sW < 32:
                continue
            scaled_rgb = cv2.resize(rgb, (sW, sH), interpolation=cv2.INTER_LINEAR)

            # Forward pass at this scale
            logits = self._get_logits(scaled_rgb, H, W)  # always resize output to (H, W)

            if accumulated_logits is None:
                accumulated_logits = logits
            else:
                accumulated_logits += logits
            count += 1

            # Horizontal flip augmentation
            if self.flip_augment:
                flipped_rgb = scaled_rgb[:, ::-1, :].copy()
                flipped_logits = self._get_logits(flipped_rgb, H, W)
                # Flip the logits back to match original orientation
                flipped_logits = torch.flip(flipped_logits, dims=[3])
                accumulated_logits += flipped_logits
                count += 1

        if accumulated_logits is None:
            return self._predict_single(rgb, H, W)

        # Average and take argmax
        avg_logits = accumulated_logits / count
        seg = avg_logits.argmax(dim=1).squeeze(0)
        return seg.cpu().numpy().astype(np.int64)

    def _predict_sliding_window(self, rgb: np.ndarray, H: int, W: int) -> np.ndarray:
        """
        Sliding window inference for very high-resolution inputs.
        Processes overlapping crops and stitches them together.
        """
        win = self.win_size
        stride = self.win_stride
        C = 19  # Cityscapes classes

        # Pad image if smaller than window
        pad_h = max(0, win - H)
        pad_w = max(0, win - W)
        if pad_h > 0 or pad_w > 0:
            rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

        pH, pW = rgb.shape[:2]

        # Accumulator tensors
        logits_sum = torch.zeros((1, C, pH, pW), dtype=torch.float32, device=self.device)
        count_map  = torch.zeros((1, 1, pH, pW), dtype=torch.float32, device=self.device)

        # Generate crop positions
        y_positions = list(range(0, pH - win + 1, stride))
        if y_positions[-1] + win < pH:
            y_positions.append(pH - win)

        x_positions = list(range(0, pW - win + 1, stride))
        if x_positions[-1] + win < pW:
            x_positions.append(pW - win)

        for y0 in y_positions:
            for x0 in x_positions:
                crop = rgb[y0:y0+win, x0:x0+win]
                crop_logits = self._get_logits(crop, win, win)
                logits_sum[:, :, y0:y0+win, x0:x0+win] += crop_logits
                count_map[:, :, y0:y0+win, x0:x0+win] += 1.0

        # Average overlapping regions
        logits_avg = logits_sum / count_map.clamp(min=1.0)

        # Crop back to original size
        logits_avg = logits_avg[:, :, :H, :W]
        seg = logits_avg.argmax(dim=1).squeeze(0)
        return seg.cpu().numpy().astype(np.int64)

    # ------------------------------------------------------------------
    # Mask extraction
    # ------------------------------------------------------------------

    def get_traversable_mask(self, seg_map: np.ndarray,
                             target_size: Tuple[int, int]) -> np.ndarray:
        """
        Build a binary traversable mask from the segmentation map.
        Applies morphological cleaning and optional edge refinement.

        Returns: HxW uint8 mask, 255 = traversable, 0 = not
        """
        H, W = target_size
        mask = np.zeros((H, W), dtype=np.uint8)

        for cls_id in TRAVERSABLE_CLASS_IDS:
            mask[seg_map == cls_id] = 255

        # Remove pixels classified as obstacles that leaked into traversable
        for cls_id in OBSTACLE_CLASS_IDS:
            mask[seg_map == cls_id] = 0

        # Morphological cleanup
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_k, self.close_k))
        kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.open_k, self.open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open)

        # Keep only the largest connected components (main road)
        mask = self._keep_largest_components(mask, n=self.top_n)

        return mask

    def get_traversable_mask_refined(self, seg_map: np.ndarray,
                                      target_size: Tuple[int, int],
                                      guide_image: np.ndarray) -> np.ndarray:
        """
        Build a traversable mask with guided-filter edge refinement.
        The guide image (original frame) helps preserve fine boundaries.

        Returns: HxW uint8 mask, 255 = traversable, 0 = not
        """
        mask = self.get_traversable_mask(seg_map, target_size)

        if self.edge_refine and guide_image is not None:
            mask = self._guided_filter_refine(mask, guide_image)

        return mask

    def _guided_filter_refine(self, mask: np.ndarray, guide: np.ndarray,
                               radius: int = 8, eps: float = 0.01) -> np.ndarray:
        """
        Apply guided filter to refine mask edges using the original image as guide.
        This aligns mask boundaries with actual image edges, producing much cleaner results.
        """
        if len(guide.shape) == 3:
            guide_gray = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY)
        else:
            guide_gray = guide

        # Normalize to [0, 1] float
        guide_f = guide_gray.astype(np.float32) / 255.0
        mask_f  = mask.astype(np.float32) / 255.0

        # Guided filter (box-filter approximation)
        mean_I  = cv2.boxFilter(guide_f, -1, (radius, radius))
        mean_p  = cv2.boxFilter(mask_f,  -1, (radius, radius))
        corr_Ip = cv2.boxFilter(guide_f * mask_f, -1, (radius, radius))
        corr_II = cv2.boxFilter(guide_f * guide_f, -1, (radius, radius))

        var_I   = corr_II - mean_I * mean_I
        cov_Ip  = corr_Ip - mean_I * mean_p

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, -1, (radius, radius))
        mean_b = cv2.boxFilter(b, -1, (radius, radius))

        refined = mean_a * guide_f + mean_b
        refined = np.clip(refined * 255, 0, 255).astype(np.uint8)

        # Re-threshold to binary
        _, refined = cv2.threshold(refined, 127, 255, cv2.THRESH_BINARY)
        return refined

    @staticmethod
    def _keep_largest_components(mask: np.ndarray, n: int = 3) -> np.ndarray:
        """Keep the N largest connected components in the mask."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask

        # Sort by area (skip label 0 = background)
        areas = stats[1:, cv2.CC_STAT_AREA]
        sorted_idx = np.argsort(areas)[::-1]
        top_n = sorted_idx[:n] + 1  # +1 to skip background label

        out = np.zeros_like(mask)
        for idx in top_n:
            out[labels == idx] = 255
        return out

    # ------------------------------------------------------------------
    # Color map for visualization
    # ------------------------------------------------------------------

    def colorize_seg_map(self, seg_map: np.ndarray) -> np.ndarray:
        """Return an HxWx3 BGR colored segmentation map."""
        H, W = seg_map.shape
        color_map = np.zeros((H, W, 3), dtype=np.uint8)

        for cls_id, color in CITYSCAPES_COLORS_BGR.items():
            color_map[seg_map == cls_id] = color

        return color_map
