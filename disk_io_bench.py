#!/usr/bin/env python3
import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

BUF_SIZE = 1024 * 1024  # 1 MiB buffer


def human_bytes(n):
    for unit in ["B/s", "KiB/s", "MiB/s", "GiB/s", "TiB/s"]:
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB/s"


def seq_write(path: str, file_size_mb: int) -> float:
    size_bytes = file_size_mb * 1024 * 1024
    buf = b"\0" * BUF_SIZE
    t0 = time.perf_counter()
    written = 0
    with open(path, "wb", buffering=0) as f:
        while written < size_bytes:
            chunk = min(BUF_SIZE, size_bytes - written)
            f.write(buf[:chunk])
            written += chunk
            # Forzar flush periódico podría sesgar; mantener sin flush para throughput bruto
    os.sync()
    return time.perf_counter() - t0


def seq_read(path: str) -> float:
    t0 = time.perf_counter()
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(BUF_SIZE)
            if not b:
                break
    os.sync()
    return time.perf_counter() - t0


def run_bench(target: str, size_mb: int, repeats: int) -> dict:
    # Archivo temporal dentro de target
    test_file = os.path.join(target, ".disk_io_bench.tmp")

    # Warmup: una escritura y lectura pequeña para calentar cachés FS
    try:
        with open(test_file, "wb") as f:
            f.write(b"warmup" * 1024)
        with open(test_file, "rb") as f:
            f.read()
    except Exception:
        pass

    write_times = []
    read_times = []
    size_bytes = size_mb * 1024 * 1024

    for _ in range(repeats):
        # Escribir
        wt = seq_write(test_file, size_mb)
        write_times.append(wt)
        # Leer
        rt = seq_read(test_file)
        read_times.append(rt)

    # Intentar limpiar caché de página (si root): no asumimos permisos, omitimos por portabilidad
    try:
        os.remove(test_file)
    except Exception:
        pass

    def agg(ts):
        return float(min(ts)), float(sum(ts) / len(ts)), float(max(ts))

    w_min, w_avg, w_max = agg(write_times)
    r_min, r_avg, r_max = agg(read_times)

    return {
        "bench": "disk_io",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "params": {
            "target": os.path.abspath(target),
            "size_mb": size_mb,
            "repeats": repeats,
        },
        "results": {
            "write": {
                "min_seconds": w_min,
                "avg_seconds": w_avg,
                "max_seconds": w_max,
                "throughput_min_bps": (size_bytes / w_max) if w_max > 0 else 0.0,
                "throughput_avg_bps": (size_bytes / w_avg) if w_avg > 0 else 0.0,
                "throughput_max_bps": (size_bytes / w_min) if w_min > 0 else 0.0,
                "throughput_avg_human": human_bytes((size_bytes / w_avg) if w_avg > 0 else 0.0),
            },
            "read": {
                "min_seconds": r_min,
                "avg_seconds": r_avg,
                "max_seconds": r_max,
                "throughput_min_bps": (size_bytes / r_max) if r_max > 0 else 0.0,
                "throughput_avg_bps": (size_bytes / r_avg) if r_avg > 0 else 0.0,
                "throughput_max_bps": (size_bytes / r_min) if r_min > 0 else 0.0,
                "throughput_avg_human": human_bytes((size_bytes / r_avg) if r_avg > 0 else 0.0),
            },
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
    p = argparse.ArgumentParser(description="Benchmark de IO de disco secuencial (write/read)")
    p.add_argument("--target", type=str, default=".", help="Directorio destino para archivo temporal")
    p.add_argument("--size-mb", type=int, default=2048, help="Tamaño de archivo a escribir/leer (MB)")
    p.add_argument("--repeats", type=int, default=3, help="Repeticiones")
    p.add_argument("--out", type=str, default="disk-io-history.json", help="Archivo de historial JSON")
    args = p.parse_args()

    os.makedirs(args.target, exist_ok=True)

    entry = run_bench(args.target, args.size_mb, args.repeats)
    print("BENCH_JSON:", json.dumps(entry, ensure_ascii=False))
    try:
        append_json(args.out, entry)
    except Exception as e:
        print("BENCH_JSON_ERROR:", str(e), file=sys.stderr)


if __name__ == "__main__":
    main()
