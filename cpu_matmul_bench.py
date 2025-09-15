#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import platform
from datetime import datetime, timezone

import numpy as np


def detect_blas_vendor():
    try:
        import numpy as np  # noqa
        cfg = np.__config__
        for name in ("openblas_info", "openblas_ilp64_info", "blas_opt_info", "mkl_info", "blis_info"):
            info = getattr(cfg, "get_info")(name)
            if info:
                return name
    except Exception:
        pass
    return "unknown"


def run_bench(size: int, dtype: str, repeats: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    dt = np.float32 if dtype == "float32" else np.float64

    # Preparar matrices
    A = rng.standard_normal((size, size), dtype=dt)
    B = rng.standard_normal((size, size), dtype=dt)

    # Warmup
    _ = A @ B

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        C = A @ B
        # Evitar optimización out: usar un valor
        s = float(C[0, 0])
        times.append(time.perf_counter() - t0)
        # Usar s para que no lo elimine el optimizador
        if s == float("nan"):
            print("", end="")

    avg = float(np.mean(times))
    min_t = float(np.min(times))
    max_t = float(np.max(times))

    # FLOPs para GEMM ~ 2*N^3
    flops = 2.0 * (size ** 3)
    gflops_avg = (flops / avg) / 1e9 if avg > 0 else 0.0
    gflops_min = (flops / max_t) / 1e9 if max_t > 0 else 0.0
    gflops_max = (flops / min_t) / 1e9 if min_t > 0 else 0.0

    return {
        "bench": "cpu_matmul",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "blas": detect_blas_vendor(),
        "params": {
            "size": size,
            "dtype": dtype,
            "repeats": repeats,
        },
        "results": {
            "avg_seconds": avg,
            "min_seconds": min_t,
            "max_seconds": max_t,
            "gflops_avg": gflops_avg,
            "gflops_min": gflops_min,
            "gflops_max": gflops_max,
        },
    }


def append_json(path: str, entry: dict):
    data = []
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
    except Exception:
        data = []
    data.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    p = argparse.ArgumentParser(description="Benchmark CPU matmul (NumPy GEMM)")
    p.add_argument("--size", type=int, default=1024, help="Tamaño de matriz NxN (default: 1024)")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32", help="Tipo de dato")
    p.add_argument("--repeats", type=int, default=5, help="Repeticiones (default: 5)")
    p.add_argument("--seed", type=int, default=1234, help="Semilla RNG (default: 1234)")
    p.add_argument("--out", type=str, default="cpu-matmul-history.json", help="Archivo de historial JSON")
    args = p.parse_args()

    entry = run_bench(args.size, args.dtype, args.repeats, args.seed)
    print("BENCH_JSON:", json.dumps(entry, ensure_ascii=False))
    try:
        append_json(args.out, entry)
    except Exception as e:
        print("BENCH_JSON_ERROR:", str(e), file=sys.stderr)


if __name__ == "__main__":
    main()
