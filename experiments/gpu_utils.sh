#!/usr/bin/env bash
# GPU polling utilities.
# Usage: source "$PROJ/experiments/gpu_utils.sh"
#        wait_for_gpu <min_free_mib> [poll_sec]

wait_for_gpu() {
    local min_free_mib=$1
    local poll_sec=${2:-60}

    # If CUDA_VISIBLE_DEVICES is pre-set to a single GPU ID, respect it and
    # only poll that specific GPU (avoids multi-job contention).
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+$ ]]; then
        local pinned_gpu="${CUDA_VISIBLE_DEVICES}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pinned to GPU ${pinned_gpu}. Waiting for >= ${min_free_mib} MiB free..."
        while true; do
            free_mib=$(nvidia-smi --query-gpu=memory.free \
                       --format=csv,noheader,nounits -i "${pinned_gpu}")
            free_mib="${free_mib// /}"
            if [ "${free_mib}" -ge "${min_free_mib}" ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${pinned_gpu} has ${free_mib} MiB free. Proceeding."
                return 0
            fi
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${pinned_gpu}: ${free_mib} MiB (need ${min_free_mib}). Retry in ${poll_sec}s..."
            sleep "$poll_sec"
        done
    fi

    # Default: auto-select the GPU with the most free memory.
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for GPU with >= ${min_free_mib} MiB free..."
    while true; do
        best=$(nvidia-smi --query-gpu=index,memory.free \
               --format=csv,noheader,nounits \
               | awk -F',' '{gsub(/ /,"",$2); print $2" "$1}' \
               | sort -rn | head -1)
        free_mib=$(echo "$best" | awk '{print $1}')
        gpu_id=$(echo  "$best" | awk '{print $2}')
        if [ "$free_mib" -ge "$min_free_mib" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Selected GPU $gpu_id (${free_mib} MiB free)."
            export CUDA_VISIBLE_DEVICES=$gpu_id
            return 0
        fi
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Best GPU $gpu_id: ${free_mib} MiB (need ${min_free_mib}). Retry in ${poll_sec}s..."
        sleep "$poll_sec"
    done
}
