"""
muBench CustomFunction: matrix_compute
========================================
Simulates a YOLO-like CNN forward pass using numpy matrix multiplications.

Each call runs a sequence of (W @ x + b) + ReLU layers, ending with a softmax
over num_classes outputs — the same compute pattern as a dense neural network
inference without PyTorch or ONNX. Designed to saturate CPU and RAM.

Interface (muBench contract):
    def matrix_compute(params: dict) -> str

params keys (all optional):
    input_size  (int)  flattened input feature length [default: 2048]
    hidden_size (int)  hidden layer width              [default: 1024]
    num_layers  (int)  number of linear layers         [default: 6]
    num_classes (int)  output class count (like YOLO 80) [default: 80]
    batch_size  (int)  virtual detection batch size    [default: 8]
"""

import numpy as np


def matrix_compute(params: dict) -> str:
    input_size  = min(int(params.get("input_size",  2048)), 8192)
    hidden_size = min(int(params.get("hidden_size", 1024)), 4096)
    num_layers  = min(int(params.get("num_layers",     6)),   12)
    num_classes = min(int(params.get("num_classes",   80)), 1000)
    batch_size  = min(int(params.get("batch_size",     8)),   64)

    rng = np.random.default_rng(seed=7)

    # Input tensor: batch_size × input_size
    x = rng.standard_normal((batch_size, input_size)).astype(np.float32)

    # Forward pass: linear layers with ReLU activations
    current_size = input_size
    for i in range(num_layers):
        next_size = hidden_size if i < num_layers - 1 else num_classes
        W = rng.standard_normal((current_size, next_size)).astype(np.float32) * 0.01
        b = rng.standard_normal((next_size,)).astype(np.float32) * 0.01
        x = np.maximum(0.0, x @ W + b)  # ReLU
        current_size = next_size

    # Softmax over class logits (numerically stable)
    x -= x.max(axis=1, keepdims=True)
    exp_x = np.exp(x)
    probs = exp_x / exp_x.sum(axis=1, keepdims=True)

    top_classes = probs.argmax(axis=1)
    top_scores  = probs.max(axis=1)

    return (
        f"matrix_compute: batch={batch_size} layers={num_layers} "
        f"input={input_size} hidden={hidden_size} classes={num_classes} "
        f"top_class={int(top_classes[0])} mean_score={float(top_scores.mean()):.4f}"
    )
