#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import platform
from datetime import datetime, timezone

import numpy as np


def run_bench(size_mb: int, repeats: int, dtype: str) -> dict:
    # Crear buffers grandes y medir copias memoria-memoria
    dt = np.uint8 if dtype == "uint8" else np.float32
    elem_size = np.dtype(dt).itemsize
    size_bytes = size_mb * 1024 * 1024
    n = size_bytes // elem_size

    src = np.zeros(n, dtype=dt)
    dst = np.zeros(n, dtype=dt)

    # warmup
    dst[:] = src[:]

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        dst[:] = src[:]
        times.append(time.perf_counter() - t0)

    avg = float(np.mean(times))
    min_t = float(np.min(times))
    max_t = float(np.max(times))

    # Ancho de banda efectivo (leer + escribir)
    bytes_moved = 2.0 * size_bytes
    gbps_avg = (bytes_moved / avg) / 1e9 if avg > 0 else 0.0
    gbps_min = (bytes_moved / max_t) / 1e9 if max_t > 0 else 0.0
    gbps_max = (bytes_moved / min_t) / 1e9 if min_t > 0 else 0.0

    return {
        "bench": "memory_bandwidth",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "params": {
            "size_mb": size_mb,
            "repeats": repeats,
            "dtype": dtype,
        },
        "results": {
            "avg_seconds": avg,
            "min_seconds": min_t,
            "max_seconds": max_t,
            "gbps_avg": gbps_avg,
            "gbps_min": gbps_min,
            "gbps_max": gbps_max,
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
    p = argparse.ArgumentParser(description="Benchmark de ancho de banda de memoria (memcpy)")
    p.add_argument("--size-mb", type=int, default=512, help="Tamaño de bloque a copiar (MB)")
    p.add_argument("--repeats", type=int, default=5, help="Repeticiones")
    p.add_argument("--dtype", choices=["uint8", "float32"], default="uint8", help="Tipo de dato base")
    p.add_argument("--out", type=str, default="memory-bandwidth-history.json", help="Archivo de historial JSON")
    args = p.parse_args()

    entry = run_bench(args.size_mb, args.repeats, args.dtype)
    print("BENCH_JSON:", json.dumps(entry, ensure_ascii=False))
    try:
        append_json(args.out, entry)
    except Exception as e:
        print("BENCH_JSON_ERROR:", str(e), file=sys.stderr)


if __name__ == "__main__":
    main()
