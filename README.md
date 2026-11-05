# roadseg

A high-performance computer vision pipeline for detecting traversable paths (roads, sidewalks, etc.) in urban environments. This system leverages the **SegFormer-B5** transformer architecture for state-of-the-art segmentation accuracy, combined with temporal smoothing and optical flow for stable video processing.

# Features

- **SegFormer-B5 Backbone**: Uses the largest SegFormer variant fine-tuned on Cityscapes for maximum precision.
- **Temporal Smoothing**: Implements optical flow (Farneback) and EMA blending to prevent mask flickering in videos.
- **Test-Time Augmentation (TTA)**: Multi-scale inference and horizontal flipping for enhanced boundary detection.
- **Edge Refinement**: Guided-filter post-processing to align path boundaries with actual image edges.
- **Real-time Preview**: Live visualization window with customizable overlays.


#  Installation

```bash
# 1. Clone and enter the project
git clone https://github.com/iMarcinn/roadseg.git
cd roadseg

# 2. Install PyTorch with CUDA (Required for GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install -r requirements.txt
```

#  Running guide

## Process a video 

```bash
python run.py --input video.mp4
```

## Process an image

```bash
python run.py --input image.jpg --output result.jpg
```

## Advanced Flags:

--preview	Show a live window while processing.

--fp16	Use half-precision to save VRAM (minor quality loss).

--no-multiscale	Disable multi-scale augmentation (much faster processing).

--no-flip	Disable horizontal flip augmentation.

--no-edge-refine	Disable guided-filter edge refinement.

--sliding-window	Use sliding window inference (required for 4K inputs).

--frame-skip N	Process every (N+1)th frame for faster video rendering.

--max-frames N	Stop processing after N frames (useful for testing).

--model [name]	Override the default SegFormer variant (e.g., nvidia/segformer-b3...).



# System Requirements

## Minimum Requirements:

GPU: NVIDIA GPU with 6GB VRAM (e.g., RTX 2060, RTX 3060).

CUDA: Version 11.8 or 12.1+.

RAM: 8GB System RAM.

Storage: ~1GB (for environment and model weights).


## Recommended Requirements (for smooth performance):

GPU: NVIDIA GPU with 8GB+ VRAM (e.g., RTX 3070, RTX 4060 Ti).

CPU: Recent multi-core processor (for optical flow calculation).

RAM: 16GB System RAM.

Note on VRAM Usage:

By default, the system is capped at 5.7GB VRAM in config.py.
If you have less than 6GB VRAM, you must use the --fp16 and --no-multiscale flags to avoid "Out of Memory" errors.

#  Tested Specifications

This system has been verified to run the full SegFormer-B5 pipeline on:

GPU: NVIDIA GeForce RTX 3050 (6GB VRAM)

RAM: 16GB System RAM

Performance: Runs optimally with the default 5.7GB VRAM cap.
