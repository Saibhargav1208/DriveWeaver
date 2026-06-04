#!/bin/bash
#SBATCH --job-name=vlm_corrected
#SBATCH --output=/data1/work/j0987341/aadya/research/DriveWeaver/logs/vlm_corrected-%j.out
#SBATCH --error=/data1/work/j0987341/aadya/research/DriveWeaver/logs/vlm_corrected-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --partition=h100
#SBATCH --signal=B:TERM@5

set -euo pipefail
set -x

########################################
# User Settings
########################################

CONTAINER=aadya_driveweaver
WORKDIR=/work

# CORRECTED VLM extraction script (dual-path: vlm_old + vlm_new)
SCRIPT_VLM=scripts/cache_vlm_nuscenes_corrected.py

VLM_PROMPT="Describe what will happen next."

VLM_ARGS=(
  --slots_path /work/data/slots/nuscenes_slots_full.pkl
  --output_dir /work/data/vlm_cache/nuscenes/
  --nuscenes_root /data/nuScenes
  --model_name Qwen/Qwen3-VL-2B-Thinking
  --layers 6 12 18 24                    # Decoder layers for pyramid guidance
  --num_keyframes 16                     # Uniform temporal sampling (was: max_frames 40)
  --resolution 384                        # Native Qwen3-VL resolution (was: 256)
  --max_new_tokens 16                    # Generate reasoning tokens (was: 1)
  --save_dtype fp16
)

RESTART_CONTAINER_ON_EXIT="true"

########################################
# Functions
########################################

log_section() {
  echo ""
  echo "========================================"
  echo "$1"
  echo "Time: $(date)"
  echo "========================================"
}

cleanup() {
  local exit_code=$?

  log_section "cleanup"

  if [[ "${RESTART_CONTAINER_ON_EXIT}" == "true" ]]; then
    echo "Restarting container: ${CONTAINER}"
    docker restart -t 10 "${CONTAINER}" 2>/dev/null || true
  fi

  echo "Container status:"
  docker ps | grep "${CONTAINER}" || true

  exit "${exit_code}"
}

get_visible_gpu_uuids() {
  nvidia-smi \
    --query-gpu=uuid \
    --format=csv,noheader \
    | paste -sd, -
}

trap cleanup TERM INT EXIT

########################################
# Job Info
########################################

log_section "Job started"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node: ${SLURM_NODELIST}"
echo "User: $(whoami)"
echo "WorkDir: ${WORKDIR}"
echo "Container: ${CONTAINER}"

########################################
# GPU Info
########################################

log_section "GPU status"

nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv

VISIBLE_UUIDS="$(get_visible_gpu_uuids)"

echo "NVIDIA_VISIBLE_DEVICES=${VISIBLE_UUIDS}"

########################################
# Container Check
########################################

log_section "Container check"

docker ps | grep "${CONTAINER}"
docker exec "${CONTAINER}" pwd

########################################
# VLM Feature Extraction (CORRECTED)
########################################

log_section "VLM Feature Extraction (CORRECTED - Dual-Path)"

echo ""
echo "========================================"
echo "CORRECTED VLM Extraction"
echo "========================================"
echo "Following ThinkJEPA official methodology:"
echo "  ✓ Dual-path: vlm_old (input) + vlm_new (reasoning)"
echo "  ✓ Per-layer pyramid: [4, 256, 3584] + [4, 16, 3584]"
echo "  ✓ Uniform temporal sampling: 16 keyframes"
echo "  ✓ Native resolution: 384x384"
echo "  ✓ Reasoning generation: 16 tokens"
echo ""
echo "Output: /work/data/vlm_cache/nuscenes/"
echo "Expected: 850 scenes (700 train + 150 val)"
echo "Size: ~7.8 MB per scene, ~6.6 GB total"
echo "Time: ~6-8 hours"
echo ""

docker exec \
  -e NVIDIA_VISIBLE_DEVICES="${VISIBLE_UUIDS}" \
  -e CUDA_VISIBLE_DEVICES="${VISIBLE_UUIDS}" \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${WORKDIR}'

    echo 'Inside container:'
    pwd
    echo 'NVIDIA_VISIBLE_DEVICES='\"\${NVIDIA_VISIBLE_DEVICES}\"
    echo 'CUDA_VISIBLE_DEVICES='\"\${CUDA_VISIBLE_DEVICES}\"

    export CUDA_VISIBLE_DEVICES=${VISIBLE_UUIDS}

    echo ''
    echo 'Starting extraction...'
    echo ''

    PYTHONPATH=${WORKDIR} python '${SCRIPT_VLM}' ${VLM_ARGS[*]} --prompt '${VLM_PROMPT}'
  "

if [ $? -ne 0 ]; then
    echo "ERROR: VLM extraction failed!"
    exit 1
fi

########################################
# Verification
########################################

log_section "Verifying extracted cache"

# Count extracted scenes
NUM_SCENES=$(docker exec "${CONTAINER}" bash -c "ls /work/data/vlm_cache/nuscenes/*.npz 2>/dev/null | wc -l")

echo "Extracted scenes: ${NUM_SCENES} / 850"

# Verify one file structure
if [ "${NUM_SCENES}" -gt 0 ]; then
    echo ""
    echo "Verifying cache structure (scene-0001.npz):"
    docker exec "${CONTAINER}" python3 -c "
import numpy as np
import sys

try:
    cache = np.load('/work/data/vlm_cache/nuscenes/scene-0001.npz')
    print('✓ Cache file loaded')
    print('  Keys:', list(cache.keys()))

    if 'vlm_old' in cache and 'vlm_new' in cache:
        print('✓ Dual-path features present')
        print('  vlm_old shape:', cache['vlm_old'].shape, '(expected: [4, 256, 3584])')
        print('  vlm_new shape:', cache['vlm_new'].shape, '(expected: [4, 16, 3584])')

        # Verify shapes
        assert cache['vlm_old'].shape == (4, 256, 3584), f'Wrong vlm_old shape: {cache[\"vlm_old\"].shape}'
        assert cache['vlm_new'].shape == (4, 16, 3584), f'Wrong vlm_new shape: {cache[\"vlm_new\"].shape}'

        print('✓ Shapes correct!')
        print('  Extraction method:', cache.get('extraction_method', 'unknown'))
        sys.exit(0)
    else:
        print('✗ Missing vlm_old or vlm_new keys!')
        sys.exit(1)
except Exception as e:
    print(f'✗ Verification failed: {e}')
    sys.exit(1)
"

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓✓✓ Cache verification PASSED ✓✓✓"
    else
        echo ""
        echo "✗✗✗ Cache verification FAILED ✗✗✗"
        exit 1
    fi
fi

########################################
# Summary
########################################

echo ""
echo "========================================"
echo "VLM Feature Extraction Complete!"
echo "End: $(date)"
echo "========================================"
echo "Extracted scenes: ${NUM_SCENES} / 850"
echo "Cache location: /work/data/vlm_cache/nuscenes/"
echo ""
echo "Cache format:"
echo "  - vlm_old: [4, 256, 3584] (input understanding)"
echo "  - vlm_new: [4, 16, 3584] (reasoning/prediction)"
echo ""
echo "Next steps:"
echo "  1. Train C-JEPA + VLM world model:"
echo "     sbatch slurm/train_cjepa_vlm.sh"
echo ""
echo "  2. Train planner (Ablation 2):"
echo "     sbatch slurm/ablation2_cjepa_vlm_planner.sh"
echo ""

EXIT_CODE=$?

trap - TERM INT EXIT
cleanup

exit "${EXIT_CODE}"
