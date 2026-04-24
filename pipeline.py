"""
Urban Path Detection System — SegFormer-only Pipeline
Architecture: SegFormer-B5 (segmentation) with multi-scale TTA + temporal smoothing.
All GPU VRAM is dedicated to path segmentation for maximum accuracy.
"""

import cv2
import numpy as np
import torch
import time
from pathlib import Path
from typing import Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

from segmentation import SegmentationModel
from visualizer import Visualizer
from temporal import TemporalSmoother


class PathDetectionPipeline:
    """
    Focused pipeline: segmentation → temporal smoothing → visualization.
    All resources dedicated to path detection accuracy.
    Processes both single images and video files.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Pipeline] Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"[Pipeline] GPU: {torch.cuda.get_device_name(0)}")
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[Pipeline] Total VRAM: {total_vram:.2f} GB")

            # Apply VRAM limit if specified
            limit_gb = self.config.get("system", {}).get("vram_limit_gb")
            if limit_gb:
                fraction = limit_gb / total_vram
                if fraction < 1.0:
                    torch.cuda.set_per_process_memory_fraction(fraction, 0)
                    print(f"[Pipeline] VRAM limit set to {limit_gb} GB ({fraction*100:.1f}% of total)")
                else:
                    print(f"[Pipeline] Requested limit {limit_gb}GB exceeds total VRAM; using full capacity.")

        print("\n[Pipeline] Loading SegFormer model...")
        self.seg_model  = SegmentationModel(config["segmentation"], self.device)
        self.visualizer = Visualizer(config["visualization"])
        self.smoother   = TemporalSmoother(config["temporal"])

        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1e6
            reserved  = torch.cuda.memory_reserved() / 1e6
            print(f"[Pipeline] VRAM used: {allocated:.0f} MB allocated, {reserved:.0f} MB reserved")

        print("[Pipeline] Model loaded — ready for path detection.\n")

    # ------------------------------------------------------------------
    # Single-frame inference
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray, prev_mask: Optional[np.ndarray] = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the full pipeline on one BGR frame.
        Returns (annotated_frame, road_mask).
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Semantic segmentation → road mask
        seg_map = self.seg_model.predict(rgb)          # (H, W) int class ids

        # 2. Build traversable mask with edge refinement
        if self.seg_model.edge_refine:
            road_mask = self.seg_model.get_traversable_mask_refined(seg_map, (h, w), rgb)
        else:
            road_mask = self.seg_model.get_traversable_mask(seg_map, (h, w))

        # 3. Temporal smoothing to eliminate flicker
        if prev_mask is not None:
            road_mask = self.smoother.smooth(road_mask, prev_mask, frame)

        # 4. Compose final annotated frame
        output = self.visualizer.draw(frame, road_mask, seg_map, self.seg_model.class_names)

        return output, road_mask

    # ------------------------------------------------------------------
    # Image processing
    # ------------------------------------------------------------------

    def process_image(self, input_path: str, output_path: Optional[str] = None) -> np.ndarray:
        frame = cv2.imread(input_path)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {input_path}")

        t0 = time.time()
        output, _ = self.process_frame(frame)
        elapsed = time.time() - t0
        print(f"[Pipeline] Image processed in {elapsed:.2f}s")

        if output_path:
            cv2.imwrite(output_path, output)
            print(f"[Pipeline] Saved to {output_path}")

        return output

    # ------------------------------------------------------------------
    # Video processing
    # ------------------------------------------------------------------

    def process_video(self, input_path: str, output_path: str,
                      frame_skip: int = 0, max_frames: Optional[int] = None):
        """
        Process a video file.
        frame_skip=0 → process every frame
        frame_skip=1 → process every other frame (interpolate between)
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")

        fps    = cap.get(cv2.CAP_PROP_FPS)
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_fps = fps / (frame_skip + 1) if frame_skip > 0 else fps

        print(f"[Pipeline] Video: {W}x{H} @ {fps:.1f}fps, {total} frames")
        print(f"[Pipeline] Frame skip: {frame_skip}, Output fps: {out_fps:.1f}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

        prev_mask   = None
        prev_output = None
        frame_idx   = 0
        processed   = 0
        t_start     = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_idx >= max_frames:
                break

            if frame_idx % (frame_skip + 1) == 0:
                output, prev_mask = self.process_frame(frame, prev_mask)
                prev_output = output
                processed += 1

                elapsed = time.time() - t_start
                eff_fps = processed / elapsed if elapsed > 0 else 0
                print(f"\r[Pipeline] Frame {frame_idx}/{total} | "
                      f"Processed: {processed} | "
                      f"Effective fps: {eff_fps:.1f}", end="", flush=True)
            else:
                # Reuse last output for skipped frames (interpolation optional)
                output = prev_output if prev_output is not None else frame

            writer.write(output)
            frame_idx += 1

        cap.release()
        writer.release()
        elapsed = time.time() - t_start
        print(f"\n[Pipeline] Done. {frame_idx} frames in {elapsed:.1f}s "
              f"({frame_idx/elapsed:.1f} output fps)")
        print(f"[Pipeline] Saved to: {output_path}")
