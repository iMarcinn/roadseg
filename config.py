"""
config.py — Central configuration for the Path Detection Pipeline.

SegFormer-only mode: all VRAM dedicated to path segmentation for maximum accuracy.
"""

CONFIG = {

    # ── System Settings ──────────────────────────────────────────────────
    "system": {
        "vram_limit_gb": 5.7,                  # Cap VRAM usage to leave room for OS/display
    },

    # ── Segmentation (SegFormer — sole model) ────────────────────────────
    "segmentation": {
        # SegFormer-B5 = largest variant, highest accuracy for Cityscapes
        # Other options (lower accuracy, faster):
        #   "nvidia/segformer-b4-finetuned-cityscapes-1024-1024"
        #   "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
        #   "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
        #   "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
        #   "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        "segformer_model": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",

        # ── Multi-scale inference (test-time augmentation) ──
        # Run inference at multiple scales and average logits for higher accuracy.
        # Costs more VRAM and time but significantly improves edge quality.
        "multi_scale": True,
        "scales": [0.75, 1.0, 1.25],         # scale factors relative to input
        "flip_augment": True,                  # also run on horizontally flipped input

        # ── Sliding window (for very high-res inputs) ──
        # Process overlapping crops instead of resizing down. Preserves fine detail.
        "use_sliding_window": False,           # enable for 4K input
        "window_size": 1024,                   # crop size in pixels
        "window_stride": 768,                  # stride (overlap = window - stride)

        # ── Post-processing ──
        "morphology_close_k": 17,              # close kernel size (fill small holes)
        "morphology_open_k": 9,                # open kernel size (remove speckles)
        "keep_top_n_components": 3,            # keep N largest connected regions
        "edge_refinement": True,               # guided-filter edge refinement

        # ── Precision ──
        "use_fp16": False,                     # set True to save VRAM (minor quality loss)
    },

    # ── Temporal Smoothing ────────────────────────────────────────────────
    "temporal": {
        # Weight of the CURRENT frame's mask (vs warped previous frame)
        # Higher = more responsive, less smoothing; Lower = smoother, more lag
        "blend_alpha":      0.75,

        # Set to False to use simple EMA blending instead of optical flow
        "use_optical_flow": True,

        # Farneback optical flow parameters
        "flow_pyr_levels":  3,
        "flow_win_size":   15,
    },

    # ── Visualization ─────────────────────────────────────────────────────
    "visualization": {
        # Path overlay color (BGR)
        "path_color_bgr":      (0, 210, 80),
        "path_edge_bgr":       (0, 255, 130),
        # Overlay transparency: 0.0 = invisible, 1.0 = opaque
        "path_alpha":          0.38,

        # Overlay semantic class colors (debug mode)
        "show_seg_classes":    False,

        # Text size for labels
        "font_scale":          0.55,
    },
}
