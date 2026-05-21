#!/usr/bin/env python3
"""
Calculate PSNR between two videos. Usage: 

python data/example-output/calc_psnr.py \
    ./data/example-output/0-wan22-i2v-svoo.mp4 \
    ./data/example-output/0-wan22-i2v-dense.mp4
"""
import argparse, subprocess, os, tempfile, math, numpy as np
from PIL import Image


def extract_frames(mp4, tmpdir):
    subprocess.run(["ffmpeg", "-y", "-i", mp4, "-vsync", "0", os.path.join(tmpdir, "%06d.png")],
                   capture_output=True)
    return sorted(os.listdir(tmpdir))


def main():
    parser = argparse.ArgumentParser(description="Calculate PSNR between two videos")
    parser.add_argument("video_a", help="Path to first video")
    parser.add_argument("video_b", nargs="?", help="Path to second video (default: same dir, 1-1.mp4 vs 1-0.mp4)")
    args = parser.parse_args()

    v0 = os.path.abspath(args.video_a)
    if args.video_b:
        v1 = os.path.abspath(args.video_b)
    else:
        root = os.path.dirname(v0)
        v1 = os.path.join(root, "1-1.mp4")

    for v in (v0, v1):
        if not os.path.isfile(v):
            raise FileNotFoundError(f"Video not found: {v}")

    with tempfile.TemporaryDirectory() as d0, tempfile.TemporaryDirectory() as d1:
        f0 = extract_frames(v0, d0)
        f1 = extract_frames(v1, d1)
        n = min(len(f0), len(f1))
        mse_sum = 0.0
        per = []
        for i in range(n):
            a = np.array(Image.open(os.path.join(d0, f0[i])), dtype=np.float64)
            b = np.array(Image.open(os.path.join(d1, f1[i])), dtype=np.float64)
            m = np.mean((a - b) ** 2)
            mse_sum += m
            if m > 0:
                per.append(10 * math.log10(255 ** 2 / m))

        mse = mse_sum / n
        psnr = 10 * math.log10(255 ** 2 / mse) if mse > 0 else float('inf')
        print(f"Frames: {len(f0)} vs {len(f1)}, compared {n}")
        print(f"PSNR: {psnr:.2f} dB  |  avg: {np.mean(per):.2f}  |  min: {np.min(per):.2f}  max: {np.max(per):.2f}")


if __name__ == "__main__":
    main()

