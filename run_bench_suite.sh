#!/usr/bin/env bash
set -euo pipefail

# Orquestador simple para correr la suite de benchmarks y producir un resumen corto.
# Todos los sub-scripts imprimen una línea que comienza con "BENCH_JSON:" seguida de un JSON.

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="python3"

KERNEL_BENCH="$ROOT_DIR/kernel-build-bench.sh"
CPU_MATMUL="$ROOT_DIR/cpu_matmul_bench.py"
MEM_BW="$ROOT_DIR/memory_bandwidth_bench.py"
DISK_IO="$ROOT_DIR/disk_io_bench.py"

OUT_DIR="${OUT_DIR:-$ROOT_DIR}"
mkdir -p "$OUT_DIR"

# Config por defecto (ajustables via env)
export MAKE_JOBS="${MAKE_JOBS:-$(nproc)}"
export TMPFS_SIZE="${TMPFS_SIZE:-6G}"
export NET_MODE="${NET_MODE:-host}"
export OUT_JSON_PATH="${OUT_JSON_PATH:-$OUT_DIR/kernel-build-history.json}"

CPU_OUT_JSON="${CPU_OUT_JSON:-$OUT_DIR/cpu-matmul-history.json}"
MEM_OUT_JSON="${MEM_OUT_JSON:-$OUT_DIR/memory-bandwidth-history.json}"
DISK_OUT_JSON="${DISK_OUT_JSON:-$OUT_DIR/disk-io-history.json}"

# Parámetros de cada bench (overridables via env)
CPU_SIZE="${CPU_SIZE:-4096}"
CPU_REPEATS="${CPU_REPEATS:-5}"
CPU_DTYPE="${CPU_DTYPE:-float32}"

MEM_SIZE_MB="${MEM_SIZE_MB:-512}"
MEM_REPEATS="${MEM_REPEATS:-5}"
MEM_DTYPE="${MEM_DTYPE:-uint8}"

DISK_TARGET="${DISK_TARGET:-$OUT_DIR}"
DISK_SIZE_MB="${DISK_SIZE_MB:-2048}"
DISK_REPEATS="${DISK_REPEATS:-3}"

summaries=()

run_and_capture() {
  local cmd="$1"
  # Ejecuta y captura la última línea BENCH_JSON:
  # shellcheck disable=SC2086
  local out
  out=$(eval $cmd | tee >(sed -n 's/^BENCH_JSON: //p' >>"$OUT_DIR/.suite_last.jsonl"))
  local json_line
  json_line=$(echo "$out" | sed -n 's/^BENCH_JSON: //p' | tail -n1)
  echo "$json_line"
}

parse_field() {
  local json="$1" key="$2"
  python3 - "$key" <<'PY'
import json,sys
k=sys.argv[1]
try:
    obj=json.load(sys.stdin)
    # Admitimos llaves anidadas simples tipo results.gbps_avg
    cur=obj
    for part in k.split('.'):
        cur=cur[part]
    if isinstance(cur,(int,float)):
        print(f"{cur}")
    else:
        print(str(cur))
except Exception:
    pass
PY
}

# Kernel build bench
if [[ -x "$KERNEL_BENCH" ]]; then
  echo "==> Kernel build bench"
  kj=$(run_and_capture "OUT_JSON_PATH=$OUT_JSON_PATH NET_MODE=$NET_MODE TMPFS_SIZE=$TMPFS_SIZE MAKE_JOBS=$MAKE_JOBS bash '$KERNEL_BENCH'")
  kver=$(parse_field "$kj" kernel_version)
  kcold=$(parse_field "$kj" cold.elapsed)
  kwarm=$(parse_field "$kj" warm.elapsed)
  summaries+=("Kernel $kver: cold=$kcold, warm=$kwarm, jobs=$MAKE_JOBS")
else
  summaries+=("Kernel: skip (script no encontrado)")
fi

# CPU matmul
if [[ -f "$CPU_MATMUL" ]]; then
  echo "==> CPU matmul bench"
  cjson=$(run_and_capture "$PY '$CPU_MATMUL' --size $CPU_SIZE --repeats $CPU_REPEATS --dtype $CPU_DTYPE --out '$CPU_OUT_JSON'")
  gflops=$(parse_field "$cjson" results.gflops_avg)
  summaries+=("CPU GEMM ${CPU_SIZE}x${CPU_SIZE}: ~${gflops} GFLOPS")
else
  summaries+=("CPU GEMM: skip (script no encontrado)")
fi

# Mem bandwidth
if [[ -f "$MEM_BW" ]]; then
  echo "==> Memory bandwidth bench"
  mjson=$(run_and_capture "$PY '$MEM_BW' --size-mb $MEM_SIZE_MB --repeats $MEM_REPEATS --dtype $MEM_DTYPE --out '$MEM_OUT_JSON'")
  mgbps=$(parse_field "$mjson" results.gbps_avg)
  summaries+=("Mem BW ${MEM_SIZE_MB}MB copy: ~${mgbps} GB/s")
else
  summaries+=("Mem BW: skip (script no encontrado)")
fi

# Disk IO
if [[ -f "$DISK_IO" ]]; then
  echo "==> Disk IO bench"
  djson=$(run_and_capture "$PY '$DISK_IO' --target '$DISK_TARGET' --size-mb $DISK_SIZE_MB --repeats $DISK_REPEATS --out '$DISK_OUT_JSON'")
  w=$(parse_field "$djson" results.write.throughput_avg_bps)
  r=$(parse_field "$djson" results.read.throughput_avg_bps)
  # Convertir a MiB/s para resumen
  to_mib() { python3 - "$1" <<'PY'
import sys
v=float(sys.argv[1])
print(f"{v/1024/1024:.1f}")
PY
}
  summaries+=("Disk IO ${DISK_SIZE_MB}MB: write=$(to_mib "$w") MiB/s, read=$(to_mib "$r") MiB/s")
else
  summaries+=("Disk IO: skip (script no encontrado)")
fi

# Resumen final
echo "\n=== Resumen de la suite ==="
for s in "${summaries[@]}"; do
  echo "- $s"
fi
