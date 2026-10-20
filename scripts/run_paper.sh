#!/bin/bash
# Every training run in the paper: 96 runs of six hours, results in results/main/.
#
#   bash scripts/run_paper.sh --dry-run    print the 96 commands
#   bash scripts/run_paper.sh 0 1 2 3      one worker per listed GPU
set -u
SCRIPT=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
cd "$(dirname "$SCRIPT")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export XLA_FLAGS=--xla_gpu_enable_command_buffer=

L=$PWD/results/main
CLAIMS=$L/.claims_paper
HOURS=6
EVALS=6
SEEDS=(997 998 999)

# cell:profile:panel
JOBS=()
for cell in kg2d:tensor_50sq_x_50 nstg40:tensor_50sq_x_50 nstg200:tensor_50sq_x_50 \
            kg3d:tensor_16cubed_x_16 ns3d:tensor_16cubed_x_16 \
            kg2d:tensor_100sq_x_100 nstg40:tensor_100sq_x_100 nstg200:tensor_100sq_x_100 \
            kg3d:tensor_25cubed_x_25 ns3d:tensor_25cubed_x_25; do
  for panel in tinn spinn r5; do
    JOBS+=("$cell:$panel")
  done
done
JOBS+=("kg2d:tensor_50sq_x_50:r32" "kg2d:tensor_50sq_x_50:every")

arm_for () {
  case $1 in
    tinn)  ARM=(--arm tinn) ;;
    spinn) ARM=(--arm spinn --layer-dims 1,64,64,64,1 --rank 32 --optimizer adam) ;;
    r5)    ARM=(--arm setinn --layer-dims 1,10,10,1 --rank 5) ;;
    r32)   ARM=(--arm setinn --layer-dims 1,10,10,1 --rank 32) ;;
    every) ARM=(--arm setinnall --layer-dims 1,10,10,1 --rank 5) ;;
  esac
}
extra_for () {
  case $1 in
    kg2d)    EXTRA=(--t-max 10) ;;
    nstg40)  EXTRA=(--nu 0.025) ;;
    nstg200) EXTRA=(--nu 0.005) ;;
    *)       EXTRA=() ;;
  esac
}
problem_for () {
  case $1 in
    nstg40|nstg200) echo "nstg" ;;
    *)              echo "$1" ;;
  esac
}
# Fill CMD with the train.py call for one job and seed.
build () {
  local cell=${1%%:*} rest=${1#*:}
  PROFILE=${rest%%:*}
  PANEL=${rest#*:}
  CELL=$cell
  arm_for "$PANEL"
  extra_for "$cell"
  OUT=$L/$cell/$PROFILE/$PANEL
  CMD=(python -u train.py --problem "$(problem_for "$cell")" --profile "$PROFILE"
       ${EXTRA[@]+"${EXTRA[@]}"} --seeds "$2" --hours "$HOURS" --evals "$EVALS"
       --skip-finished --out "$OUT" "${ARM[@]}")
}

worker () {
  local gpu=$1
  for seed in "${SEEDS[@]}"; do
    for job in "${JOBS[@]}"; do
      mkdir "$CLAIMS/${job//:/.}.$seed" 2>/dev/null || continue
      build "$job" "$seed"
      mkdir -p "$OUT"
      echo "[GPU $gpu] $CELL / $PROFILE / $PANEL / seed $seed  $(date '+%m-%d %H:%M')"
      CUDA_VISIBLE_DEVICES=$gpu "${CMD[@]}" >> "$L/$CELL-$PROFILE-$PANEL-s$seed.log" 2>&1
    done
  done
  echo "[GPU $gpu] queue empty  $(date '+%m-%d %H:%M')"
}

if [ "${1:-}" = "--dry-run" ]; then
  for seed in "${SEEDS[@]}"; do
    for job in "${JOBS[@]}"; do
      build "$job" "$seed"
      echo "${CMD[*]}"
    done
  done
  exit 0
fi
if [ "${1:-}" = "--worker" ]; then
  worker "$2"
  exit 0
fi
if [ $# -lt 1 ]; then
  echo "usage: bash scripts/run_paper.sh --dry-run | bash scripts/run_paper.sh GPU [GPU ...]" >&2
  exit 2
fi
if pgrep -f "train.py --problem" > /dev/null; then
  echo "train.py is already running on this machine; stop it first." >&2
  exit 1
fi
mkdir -p "$L"
rm -rf "$CLAIMS"
mkdir -p "$CLAIMS"
for gpu in "$@"; do
  nohup bash "$SCRIPT" --worker "$gpu" > "$L/gpu$gpu-paper.log" 2>&1 &
done
n=$(( ${#JOBS[@]} * ${#SEEDS[@]} ))
echo "$# worker(s) on $n jobs, ${HOURS} h each ($(( n * HOURS )) GPU-hours)"
echo "watch with:  python tools/sweep_status.py"
