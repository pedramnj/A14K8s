"""
muBench CustomFunction: image_processing
========================================
Simulates a synthetic image decode → 2D convolution filter → histogram
equalization → encode pipeline using only numpy.

Interface (muBench contract):
    def image_processing(params: dict) -> str

params keys (all optional):
    width      (int)  image width  in pixels   [default: 640]
    height     (int)  image height in pixels   [default: 480]
    iterations (int)  convolution filter passes [default: 3]
"""

import numpy as np


def image_processing(params: dict) -> str:
    width      = min(int(params.get("width",      640)), 1920)
    height     = min(int(params.get("height",     480)), 1080)
    iterations = min(int(params.get("iterations",   3)),    8)

    rng = np.random.default_rng(seed=42)

    # Simulate decode: allocate a synthetic RGB frame (float32)
    frame = rng.integers(0, 256, (height, width, 3), dtype=np.uint8).astype(np.float32)

    # 3×3 Gaussian-like kernel
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float32) / 16.0

    # Simulate per-channel blurring via manual sliding-window convolution.
    # Deliberately unoptimised (no scipy) to keep CPU load high.
    for _ in range(iterations):
        for c in range(3):
            ch = frame[:, :, c]
            out = np.zeros_like(ch)
            for ky in range(3):
                for kx in range(3):
                    out[1:-1, 1:-1] += (
                        ch[ky:height - 2 + ky, kx:width - 2 + kx] * kernel[ky, kx]
                    )
            frame[:, :, c] = out

    # Simulate histogram equalization (heavy cumsum path)
    gray = frame.mean(axis=2)
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0.0, 255.0))
    cdf = hist.cumsum().astype(np.float32)
    cdf_min = cdf[cdf > 0].min()
    total = float(height * width)
    eq_map = ((cdf - cdf_min) / max(total - cdf_min, 1.0) * 255.0).clip(0, 255)
    _ = eq_map[gray.astype(np.int32).clip(0, 255)]

    return (
        f"image_processing: size={width}x{height} passes={iterations} "
        f"mean={frame.mean():.2f} std={frame.std():.2f}"
    )
