#!/bin/bash
# Profile the SM90 MegaMoE kernel with shared experts enabled.
#
# Goal: find where the shared phase spends time. WarpStateStats gives the stall
# reason breakdown; if the shared mainloop's per-k-block warpgroup_wait<0> is the
# bottleneck, stalls concentrate on MMA/barrier waits.
#
# Usage: run_ncu_sm90.sh <tokens> <ns> <outdir>
set -e

tokens=${1:-2048}
ns=${2:-1}
outdir=${3:-/work/ncu_out}
num_processes=8

mkdir -p "$outdir"
export DG_JIT_WITH_LINEINFO=1
export EP_DISABLE_GIN=1
export PYTHONPATH=/work/DeepGEMM-dev

cd /work/DeepGEMM-dev

echo "=== Warm up JIT cache (tokens=$tokens ns=$ns) ==="
python3 tests/bench_shared_ncu.py --ncu-profile-only \
    --tokens "$tokens" --ns "$ns" --num-processes $num_processes || true

sleep 2

ncu_args=(
    --config-file off
    --force-overwrite
    --kernel-name sm90_fp8_mega_moe_impl
    --replay-mode application
    --section WarpStateStats
    --section SpeedOfLight
    --section MemoryWorkloadAnalysis
    --launch-skip 0
    --launch-count 1
    --lockstep-kernel-launch
    --communicator tcp
    --clock-control none
    --communicator-tcp-num-peers "$num_processes"
    --kill yes
    --app-replay-buffer memory
)

echo "=== Profile ==="
for ((i = 0; i < num_processes; ++i)); do
    ncu "${ncu_args[@]}" -o "${outdir%/}/sm90-shared-t${tokens}-ns${ns}.$i" \
        python3 tests/bench_shared_ncu.py \
            --local-rank-idx=$i \
            --ncu-profile-only \
            --tokens "$tokens" --ns "$ns" \
            --num-processes $num_processes &
done

wait
echo "=== Done: $outdir ==="
