#!/usr/bin/env bash
# Kernel build benchmark in an ephemeral container (Docker/Podman), RAM-only workspace.
# Usage:
#   bash kernel-build-bench.sh            # defaults: gcc + linux-6.10.9
#   KVER=6.10.9 CC_KIND=gcc bash kernel-build-bench.sh
#   KVER=6.10.9 CC_KIND=clang bash kernel-build-bench.sh

set -euo pipefail

# ---- Config (override via env) ----
KVER="${KVER:-6.10.9}"          # Pin a specific kernel version
CC_KIND="${CC_KIND:-gcc}"       # gcc | clang
IMAGE="ubuntu:24.04"            # Base toolchain image
# Networking and tmpfs defaults (tune via env)
NET_MODE="${NET_MODE:-host}"    # host | bridge | none (default host to avoid DNS hiccups)
DNS_SERVERS=${DNS_SERVERS:-"1.1.1.1 8.8.8.8"}  # used only if NET_MODE!=host
TMPFS_SIZE="${TMPFS_SIZE:-16G}" # tmpfs size for RAM workspace
MAKE_JOBS="${MAKE_JOBS:-$(nproc)}"
# Output JSON on host (history). Default: ./kernel-build-history.json
OUT_JSON_PATH="${OUT_JSON_PATH:-$PWD/kernel-build-history.json}"
OUT_JSON_BASENAME="$(basename "$OUT_JSON_PATH")"

# ---- Pick container runtime ----
RUNTIME=""
if command -v podman >/dev/null 2>&1; then RUNTIME="podman"
elif command -v docker >/dev/null 2>&1; then RUNTIME="docker"
else
  echo "Error: install Docker or Podman first." >&2
  exit 1
fi

# ---- Compose tool install list ----
APT="build-essential bc bison flex libssl-dev libelf-dev dwarves libncurses-dev xz-utils curl ca-certificates time ccache"
if [[ "$CC_KIND" == "clang" ]]; then
  APT="$APT clang lld"
fi
# For JSON serialization at the end
APT="$APT python3"

# ---- Run ephemeral build ----
# - --rm: auto-remove container
# - --mount type=tmpfs,destination=/work: everything in RAM
# - nothing persists on host; only the base image layer may be cached locally
"$RUNTIME" run --rm \
  $(
    # Build networking opts per runtime
    if [[ "$NET_MODE" == host ]]; then
      if [[ "$RUNTIME" == "docker" ]]; then echo --network=host; else echo --network host; fi
    elif [[ "$NET_MODE" == bridge ]]; then
      # default bridge network; optionally inject DNS servers
      if [[ -n "${DNS_SERVERS:-}" ]]; then
        for d in $DNS_SERVERS; do echo --dns $d; done
      fi
    elif [[ "$NET_MODE" == none ]]; then
      echo --network none
    fi
  ) \
  --tmpfs /work:exec,size=$TMPFS_SIZE \
  --mount type=bind,source="$(pwd)",destination=/host \
  -e http_proxy -e https_proxy -e no_proxy -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY \
  -e KVER="$KVER" -e CC_KIND="$CC_KIND" -e MAKE_JOBS="$MAKE_JOBS" -e NET_MODE="$NET_MODE" -e TMPFS_SIZE="$TMPFS_SIZE" -e IMAGE="$IMAGE" \
  -e OUT_JSON_PATH="/host/$OUT_JSON_BASENAME" \
  "$IMAGE" bash -lc "
set -euo pipefail

# 1) Toolchain
export DEBIAN_FRONTEND=noninteractive
APT_OPTS='-o Acquire::Retries=3 -o Acquire::ForceIPv4=true -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30'
apt-get \$APT_OPTS -qq update
apt-get \$APT_OPTS -yqq install $APT

# 2) Fetch kernel source (specific, reproducible)
cd /work
curl -sSLO https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-${KVER}.tar.xz
tar -xf linux-${KVER}.tar.xz
cd linux-${KVER}

# 3) Out-of-tree build dir + optional ccache
mkdir -p /work/build /work/ccache
export CCACHE_DIR=/work/ccache
ccache -M 10G >/dev/null 2>&1 || true
ccache -z >/dev/null 2>&1 || true

if [[ "\$CC_KIND" = "clang" ]]; then
  export CC='ccache clang'
  export LD=ld.lld
else
  export CC='ccache gcc'
fi

# 4) Configure (defconfig = standard baseline)
make O=/work/build defconfig

echo '--- COLD BUILD (no ccache) ---'
COLD_LOG=/work/cold_time.txt
/usr/bin/time -v make O=/work/build -j$MAKE_JOBS 2> >(tee "\$COLD_LOG" >&2)

echo
echo 'ccache stats after cold build:'
ccache -s || true

echo
echo '--- WARM BUILD (ccache-on from previous pass) ---'
# Clean object files but keep source; ccache will accelerate recompilation.
make O=/work/build clean
WARM_LOG=/work/warm_time.txt
/usr/bin/time -v make O=/work/build -j$MAKE_JOBS 2> >(tee "\$WARM_LOG" >&2)

echo
echo 'ccache stats after warm build:'
ccache -s || true

echo
echo '=== SUMMARY ==='
cold_elapsed=\$(grep -F 'Elapsed (wall clock) time' "\$COLD_LOG" | awk -F': ' '{print \$2}' | tail -n1)
cold_cpu=\$(grep -F 'Percent of CPU this job got' "\$COLD_LOG" | awk -F': ' '{print \$2}' | tail -n1)
cold_rss=\$(grep -F 'Maximum resident set size' "\$COLD_LOG" | awk -F': ' '{print \$2}' | tail -n1)
warm_elapsed=\$(grep -F 'Elapsed (wall clock) time' "\$WARM_LOG" | awk -F': ' '{print \$2}' | tail -n1)
warm_cpu=\$(grep -F 'Percent of CPU this job got' "\$WARM_LOG" | awk -F': ' '{print \$2}' | tail -n1)
warm_rss=\$(grep -F 'Maximum resident set size' "\$WARM_LOG" | awk -F': ' '{print \$2}' | tail -n1)
printf 'Kernel %s | CC: %s | Jobs: %s\n' "\$KVER" "\$CC_KIND" "\$MAKE_JOBS"
printf 'Cold build:  elapsed=%s, CPU=%s, MaxRSS=%sKB\n' "\$cold_elapsed" "\$cold_cpu" "\$cold_rss"
printf 'Warm build:  elapsed=%s, CPU=%s, MaxRSS=%sKB\n' "\$warm_elapsed" "\$warm_cpu" "\$warm_rss"
export cold_elapsed="\$cold_elapsed" cold_cpu="\$cold_cpu" cold_rss="\$cold_rss"
export warm_elapsed="\$warm_elapsed" warm_cpu="\$warm_cpu" warm_rss="\$warm_rss"
echo
# Persist JSON history on host via /host mount
python3 - <<'PY'
import json
import os
import time
import traceback

def parse_elapsed(s):
  if not s:
    return None
  parts = s.strip().split(':')
  try:
    if len(parts) == 3:
      h = int(parts[0]); m = int(parts[1]); sec = float(parts[2])
      return h*3600 + m*60 + sec
    if len(parts) == 2:
      m = int(parts[0]); sec = float(parts[1])
      return m*60 + sec
    return float(parts[0])
  except Exception:
    return None

try:
  # Campos base desde entorno
  timestamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
  kver = os.environ.get('KVER')
  cc_kind = os.environ.get('CC_KIND')
  jobs = int(os.environ.get('MAKE_JOBS', '0') or 0)
  net_mode = os.environ.get('NET_MODE')
  tmpfs_size = os.environ.get('TMPFS_SIZE')
  image = os.environ.get('IMAGE', 'ubuntu:24.04')

  cold_elapsed = os.environ.get('cold_elapsed')
  warm_elapsed = os.environ.get('warm_elapsed')

  entry = {
    'timestamp': timestamp,
    'kernel_version': kver,
    'cc_kind': cc_kind,
    'jobs': jobs,
    'net_mode': net_mode,
    'tmpfs_size': tmpfs_size,
    'image': image,
    'cold': {
      'elapsed': cold_elapsed,
      'elapsed_seconds': parse_elapsed(cold_elapsed),
      'cpu': os.environ.get('cold_cpu'),
      'max_rss_kb': os.environ.get('cold_rss'),
    },
    'warm': {
      'elapsed': warm_elapsed,
      'elapsed_seconds': parse_elapsed(warm_elapsed),
      'cpu': os.environ.get('warm_cpu'),
      'max_rss_kb': os.environ.get('warm_rss'),
    },
  }

  out_path = os.environ.get('OUT_JSON_PATH', '/host/kernel-build-history.json')
  # Apéndice a archivo JSON tipo array
  if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
    with open(out_path, 'r+', encoding='utf-8') as f:
      try:
        data = json.load(f)
        if not isinstance(data, list):
          data = []
      except Exception:
        data = []
      data.append(entry)
      f.seek(0)
      json.dump(data, f, indent=2)
      f.truncate()
  else:
    with open(out_path, 'w', encoding='utf-8') as f:
      json.dump([entry], f, indent=2)

  print('BENCH_JSON:', json.dumps(entry, separators=(',', ':')))
except Exception as e:
  print('BENCH_JSON_ERROR:', str(e) + '\n' + traceback.format_exc())
PY

echo 'DONE. All artifacts live in tmpfs and vanish when the container exits.'
"

echo
echo "Tip: to remove the pulled base image as well (leave ZERO residue), run:"
if [[ "$RUNTIME" == "podman" ]]; then
  echo "  podman rmi $IMAGE >/dev/null 2>&1 || true"
else
  echo "  docker rmi $IMAGE >/dev/null 2>&1 || true"
fi
