"""Inference latency benchmark (ms per URL).

The whole rationale for choosing DistilBERT over BERT is speed for real-time
use. This script measures that claim: it times single-URL inference on CPU
and, if available, GPU, reporting mean and p95 latency after a warm-up.

Usage (from the repository root):
    python -m src.benchmark
    python -m src.benchmark --n 500
"""

import argparse
import time

import numpy as np
import torch

from .predict import load_model, predict_url

# A small pool of representative URLs to time against.
SAMPLE_URLS = [
    "https://www.google.com",
    "http://secure-login.paypa1-account.example/verify",
    "http://192.168.0.1/wp-admin/setup-config.php",
    "https://github.com/najla-jaashan/GP-Malicious-URL-Detection",
    "http://free-prize-claim.ru/win?id=8827",
]


def _time_device(tokenizer, model, device: str, n: int) -> dict:
    """Return latency stats (ms) for ``n`` single-URL predictions on device."""
    model.to(device)

    # Warm-up: the first few calls include lazy CUDA/graph init overhead.
    for url in SAMPLE_URLS:
        predict_url(url, tokenizer, model)

    timings = []
    for i in range(n):
        url = SAMPLE_URLS[i % len(SAMPLE_URLS)]
        start = time.perf_counter()
        predict_url(url, tokenizer, model)
        if device == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)  # -> ms

    arr = np.array(timings)
    return {
        "device": device,
        "n": n,
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "throughput_per_s": float(1000.0 / arr.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Latency benchmark")
    parser.add_argument("--n", type=int, default=200, help="requests per device")
    args = parser.parse_args()

    tokenizer, model = load_model()

    results = [_time_device(tokenizer, model, "cpu", args.n)]
    if torch.cuda.is_available():
        results.append(_time_device(tokenizer, model, "cuda", args.n))
    else:
        print("(CUDA not available -- reporting CPU only.)")

    print(
        f"\n{'Device':<8}{'Mean(ms)':>10}{'p50(ms)':>10}{'p95(ms)':>10}{'URLs/s':>10}"
    )
    print("-" * 48)
    for r in results:
        print(
            f"{r['device']:<8}{r['mean_ms']:>10.2f}{r['p50_ms']:>10.2f}"
            f"{r['p95_ms']:>10.2f}{r['throughput_per_s']:>10.1f}"
        )


if __name__ == "__main__":
    main()
