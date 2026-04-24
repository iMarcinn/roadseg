"""
run.py — Entry point for the Urban Path Detection System.

Usage examples:
    # Process an image
    python run.py --input street.jpg --output result.jpg

    # Process a video
    python run.py --input dashcam.mp4 --output result.mp4

    # Process video, skip every other frame (faster)
    python run.py --input dashcam.mp4 --output result.mp4 --frame-skip 1

    # Disable multi-scale TTA (faster, less accurate)
    python run.py --input street.jpg --no-multiscale

    # Use FP16 precision (saves VRAM, minor quality loss)
    python run.py --input street.jpg --fp16

    # Choose a different SegFormer variant
    python run.py --input street.jpg --model nvidia/segformer-b3-finetuned-cityscapes-1024-1024

    # Webcam real-time (experimental)
    python run.py --webcam

    # Show live preview window while processing video
    python run.py --input dashcam.mp4 --output result.mp4 --preview
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

from config import CONFIG
from pipeline import PathDetectionPipeline


def parse_args():
    p = argparse.ArgumentParser(
        description="Urban Path Detection — SegFormer-B5 (maximum accuracy)"
    )
    p.add_argument("--input",       "-i", type=str, help="Input image or video file")
    p.add_argument("--output",      "-o", type=str, help="Output file path")
    p.add_argument("--frame-skip",        type=int, default=0,
                   help="Skip N frames between processed frames (0 = process all)")
    p.add_argument("--max-frames",        type=int, default=None,
                   help="Stop after N frames (useful for testing)")
    p.add_argument("--model",             type=str, default=None,
                   help="Override SegFormer model (e.g. nvidia/segformer-b3-finetuned-cityscapes-1024-1024)")
    p.add_argument("--no-multiscale",     action="store_true",
                   help="Disable multi-scale test-time augmentation (faster)")
    p.add_argument("--no-flip",           action="store_true",
                   help="Disable horizontal flip augmentation")
    p.add_argument("--no-edge-refine",    action="store_true",
                   help="Disable guided-filter edge refinement")
    p.add_argument("--fp16",              action="store_true",
                   help="Use FP16 precision (saves VRAM, minor quality loss)")
    p.add_argument("--sliding-window",    action="store_true",
                   help="Use sliding window inference (for 4K+ inputs)")
    p.add_argument("--preview",           action="store_true",
                   help="Show live preview window during processing")
    p.add_argument("--webcam",            action="store_true",
                   help="Use webcam as input (device 0)")
    return p.parse_args()


def build_config(args) -> dict:
    import copy
    cfg = copy.deepcopy(CONFIG)

    if args.model:
        cfg["segmentation"]["segformer_model"] = args.model
    if args.no_multiscale:
        cfg["segmentation"]["multi_scale"] = False
    if args.no_flip:
        cfg["segmentation"]["flip_augment"] = False
    if args.no_edge_refine:
        cfg["segmentation"]["edge_refinement"] = False
    if args.fp16:
        cfg["segmentation"]["use_fp16"] = True
    if args.sliding_window:
        cfg["segmentation"]["use_sliding_window"] = True

    return cfg


def is_video(path: str) -> bool:
    return Path(path).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def run_webcam(pipeline: PathDetectionPipeline):
    """Real-time webcam mode."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Cannot open webcam")
        sys.exit(1)

    print("[run] Webcam mode — press Q to quit")
    prev_mask = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output, prev_mask = pipeline.process_frame(frame, prev_mask)

        cv2.imshow("Path Detection", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()

    if not args.input and not args.webcam:
        print("Error: provide --input <file> or --webcam")
        sys.exit(1)

    cfg = build_config(args)
    pipeline = PathDetectionPipeline(cfg)

    if args.webcam:
        run_webcam(pipeline)
        return

    input_path = args.input

    if is_video(input_path):
        # ── Video mode ──
        if not args.output:
            stem = Path(input_path).stem
            args.output = f"{stem}_path_detection.mp4"
            print(f"[run] No output path given, saving to: {args.output}")

        if args.preview:
            _process_video_with_preview(pipeline, input_path, args.output,
                                        args.frame_skip, args.max_frames)
        else:
            pipeline.process_video(input_path, args.output,
                                   args.frame_skip, args.max_frames)

    else:
        # ── Image mode ──
        if not args.output:
            stem = Path(input_path).stem
            ext  = Path(input_path).suffix
            args.output = f"{stem}_path_detection{ext}"
            print(f"[run] No output path given, saving to: {args.output}")

        output = pipeline.process_image(input_path, args.output)

        if args.preview:
            cv2.imshow("Path Detection", output)
            print("[run] Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def _process_video_with_preview(pipeline, input_path, output_path,
                                 frame_skip, max_frames):
    """Process video while showing a live preview window."""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    prev_mask   = None
    prev_output = None
    frame_idx   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        if frame_idx % (frame_skip + 1) == 0:
            output, prev_mask = pipeline.process_frame(frame, prev_mask)
            prev_output = output
        else:
            output = prev_output if prev_output is not None else frame

        writer.write(output)
        cv2.imshow("Path Detection (press Q to stop)", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"[run] Saved to: {output_path}")


if __name__ == "__main__":
    main()
