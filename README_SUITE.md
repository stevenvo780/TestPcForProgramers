# 🧪 Suite de Benchmarks del Sistema

Incluye varios benchmarks con salida consistente en JSON y un orquestador:

- `kernel-build-bench.sh`: Compila el kernel Linux (runs en frío y caliente) y guarda resultados en `kernel-build-history.json`.
- `cpu_matmul_bench.py`: GEMM con NumPy; mide GFLOPS y guarda en `cpu-matmul-history.json`.
- `memory_bandwidth_bench.py`: Copia memoria-memoria; estima GB/s y guarda en `memory-bandwidth-history.json`.
- `disk_io_bench.py`: Escritura/lectura secuencial; reporta MiB/s y guarda en `disk-io-history.json`.
- `run_bench_suite.sh`: Ejecuta todos los anteriores y muestra un resumen.

## Uso rápido

```bash
# Kernel build bench (por defecto usa todos los núcleos y tmpfs 6G)
OUT_JSON_PATH=kernel-build-history.json \
NET_MODE=host TMPFS_SIZE=6G MAKE_JOBS=$(nproc) \
bash kernel-build-bench.sh

# CPU GEMM
python3 cpu_matmul_bench.py --size 4096 --repeats 5 --dtype float32 --out cpu-matmul-history.json

# Memoria (copy)
python3 memory_bandwidth_bench.py --size-mb 512 --repeats 5 --dtype uint8 --out memory-bandwidth-history.json

# Disco (secuencial)
python3 disk_io_bench.py --target . --size-mb 2048 --repeats 3 --out disk-io-history.json

# Suite completa (con resumen)
OUT_DIR=. bash run_bench_suite.sh
```

Cada script imprime una línea `BENCH_JSON: {...}` y, además, agrega la entrada a su archivo de historial correspondiente.
