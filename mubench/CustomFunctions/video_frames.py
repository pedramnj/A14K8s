"""
muBench CustomFunction: video_frames
======================================
Simulates a video processing pipeline: decode → motion detection (frame diff)
→ optical flow approximation (Sobel gradients) → per-frame FFT energy (encode).

Each call processes num_frames synthetic grayscale frames. The multi-frame loop
with per-frame numpy ops creates sustained CPU pressure over the full request
duration — ideal for generating meaningful autoscaling signals.

Interface (muBench contract):
    def video_frames(params: dict) -> str

params keys (all optional):
    num_frames        (int)   frames to process         [default: 16]
    frame_width       (int)   frame width  in pixels    [default: 320]
    frame_height      (int)   frame height in pixels    [default: 240]
    motion_threshold  (float) pixel-diff motion trigger [default: 25.0]
"""

import numpy as np


def video_frames(params: dict) -> str:
    num_frames       = min(int(params.get("num_frames",   16)),   64)
    frame_width      = min(int(params.get("frame_width",  320)), 1280)
    frame_height     = min(int(params.get("frame_height", 240)),  720)
    motion_threshold = float(params.get("motion_threshold", 25.0))

    rng = np.random.default_rng(seed=13)

    # Generate synthetic grayscale frames with slow translational motion
    frames = rng.integers(
        0, 256, (num_frames, frame_height, frame_width), dtype=np.uint8
    ).astype(np.float32)

    for i in range(1, num_frames):
        sx, sy = int(rng.integers(-5, 6)), int(rng.integers(-5, 6))
        frames[i] = np.roll(np.roll(frames[i], sy, axis=0), sx, axis=1)

    motion_events  = 0
    total_flow_mag = 0.0
    fft_energies   = []

    prev = frames[0]
    for i in range(1, num_frames):
        curr = frames[i]

        # Motion detection: frame difference
        diff = np.abs(curr - prev)
        if int((diff > motion_threshold).sum()) > (frame_width * frame_height * 0.01):
            motion_events += 1

        # Optical flow approximation via Sobel-like gradients on the diff image
        gx = diff[:, 1:] - diff[:, :-1]
        gy = diff[1:, :] - diff[:-1, :]
        mh = min(gx.shape[0], gy.shape[0])
        mw = min(gx.shape[1], gy.shape[1])
        total_flow_mag += float(np.sqrt(gx[:mh, :mw] ** 2 + gy[:mh, :mw] ** 2).mean())

        # DCT-like frequency energy via FFT (first 4 frames only)
        if i <= 4:
            fft_energies.append(float(np.abs(np.fft.fft2(curr)).mean()))

        prev = curr

    avg_flow   = total_flow_mag / max(1, num_frames - 1)
    avg_energy = float(np.mean(fft_energies)) if fft_energies else 0.0

    return (
        f"video_frames: frames={num_frames} size={frame_width}x{frame_height} "
        f"motion_events={motion_events} avg_flow={avg_flow:.3f} "
        f"fft_energy={avg_energy:.2f}"
    )
