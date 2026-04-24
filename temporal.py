"""
Temporal Smoother — optical flow based mask blending.
Eliminates frame-to-frame flicker in the road segmentation mask
without requiring any additional ML model.
"""

import numpy as np
import cv2
from typing import Optional


class TemporalSmoother:

    def __init__(self, config: dict):
        self.blend_alpha   = config.get("blend_alpha", 0.75)   # weight of current frame
        self.use_flow      = config.get("use_optical_flow", True)
        self.prev_gray     = None
        self.flow_levels   = config.get("flow_pyr_levels", 3)
        self.flow_winsize  = config.get("flow_win_size", 15)

    def smooth(self,
               curr_mask: np.ndarray,
               prev_mask: np.ndarray,
               curr_frame: np.ndarray) -> np.ndarray:
        """
        Blend current mask with optical-flow-warped previous mask.

        Args:
            curr_mask:  HxW uint8 current frame's road mask
            prev_mask:  HxW uint8 previous frame's road mask
            curr_frame: HxWx3 BGR current frame (for optical flow)

        Returns:
            smoothed_mask: HxW uint8
        """
        if not self.use_flow:
            # Simple exponential moving average (no flow)
            blended = cv2.addWeighted(
                curr_mask.astype(np.float32), self.blend_alpha,
                prev_mask.astype(np.float32), 1.0 - self.blend_alpha,
                0
            )
            return (blended > 127).astype(np.uint8) * 255

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return curr_mask

        # Compute dense optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, curr_gray,
            None,
            pyr_scale=0.5,
            levels=self.flow_levels,
            winsize=self.flow_winsize,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Warp previous mask using the flow field
        warped_prev = self._warp_mask(prev_mask, flow)

        # Blend warped previous with current
        alpha = self.blend_alpha
        blended = alpha * curr_mask.astype(np.float32) + (1.0 - alpha) * warped_prev.astype(np.float32)
        smoothed = (blended > 127).astype(np.uint8) * 255

        self.prev_gray = curr_gray
        return smoothed

    @staticmethod
    def _warp_mask(mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp a binary mask by a dense optical flow field."""
        H, W = mask.shape[:2]
        # Create mesh grid
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        # Apply flow displacement
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        # Remap
        warped = cv2.remap(
            mask, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        return warped

    def reset(self):
        """Call between videos or when processing images."""
        self.prev_gray = None
