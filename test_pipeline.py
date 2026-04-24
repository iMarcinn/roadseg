"""
test_pipeline.py — Smoke test: verify SegFormer loads and runs end-to-end.
Run this BEFORE testing on real footage to catch import/CUDA errors early.

Usage:
    python test_pipeline.py
    python test_pipeline.py --no-multiscale    # faster test
    python test_pipeline.py --fp16             # test FP16 mode
"""

import argparse
import time
import numpy as np
import cv2
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-multiscale", action="store_true",
                        help="Disable multi-scale TTA for faster testing")
    parser.add_argument("--fp16", action="store_true",
                        help="Test with FP16 precision")
    parser.add_argument("--save", default="test_output.jpg",
                        help="Save annotated test image to this path")
    args = parser.parse_args()

    print("=" * 60)
    print("Path Detection — Smoke Test (SegFormer-only)")
    print("=" * 60)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram:.1f} GB")

    import copy
    from config import CONFIG
    from pipeline import PathDetectionPipeline

    cfg = copy.deepcopy(CONFIG)
    if args.no_multiscale:
        cfg["segmentation"]["multi_scale"] = False
    if args.fp16:
        cfg["segmentation"]["use_fp16"] = True

    ms_status = "OFF" if args.no_multiscale else "ON"
    fp_status = "FP16" if args.fp16 else "FP32"
    print(f"\nMulti-scale TTA: {ms_status} | Precision: {fp_status}")

    # ── Create a synthetic test frame (simulated street scene colors) ──
    H, W = 720, 1280
    frame = np.zeros((H, W, 3), dtype=np.uint8)

    # Sky (top 35%)
    frame[:int(H*0.35), :] = [180, 120, 80]

    # Buildings (sides, 35-75%)
    frame[int(H*0.35):int(H*0.75), :int(W*0.2)]   = [100, 100, 110]
    frame[int(H*0.35):int(H*0.75), int(W*0.8):]   = [100, 100, 110]

    # Road (bottom 50%, gray)
    frame[int(H*0.5):, :] = [80, 80, 80]

    # Lane markings
    for y in range(int(H*0.55), H, 60):
        cv2.rectangle(frame, (W//2 - 5, y), (W//2 + 5, y + 30), (240, 240, 240), -1)

    # Simulated car
    cv2.rectangle(frame, (int(W*0.6), int(H*0.55)), (int(W*0.75), int(H*0.72)),
                  (40, 60, 180), -1)
    # Simulated pedestrian
    cv2.rectangle(frame, (int(W*0.3), int(H*0.52)), (int(W*0.32), int(H*0.72)),
                  (200, 120, 80), -1)

    print("\n[Test] Running pipeline on synthetic 1280x720 frame...")
    t0 = time.time()
    pipeline = PathDetectionPipeline(cfg)
    load_time = time.time() - t0
    print(f"[Test] Model load time: {load_time:.1f}s")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e6
        reserved  = torch.cuda.memory_reserved() / 1e6
        print(f"[Test] VRAM: {allocated:.0f} MB allocated, {reserved:.0f} MB reserved")

    # Warmup
    print("[Test] Warmup run...")
    _, _ = pipeline.process_frame(frame.copy())

    # Timed run
    print("[Test] Timed run...")
    t1 = time.time()
    output, mask = pipeline.process_frame(frame.copy())
    inf_time = time.time() - t1

    print(f"\n[Test] Inference time: {inf_time:.2f}s ({1/inf_time:.1f} fps)")
    print(f"[Test] Road mask coverage: {mask.sum() / (255*H*W)*100:.1f}%")
    print(f"[Test] Output shape: {output.shape}")

    if args.save:
        cv2.imwrite(args.save, output)
        print(f"[Test] Saved output to: {args.save}")

    print("\n[Test] ✓ All systems nominal. Pipeline is ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
