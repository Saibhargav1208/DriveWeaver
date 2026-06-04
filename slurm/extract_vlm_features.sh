#!/bin/bash
#SBATCH --job-name=vlm_extract
#SBATCH --output=/data1/work/j0987341/aadya/research/DriveWeaver/logs/vlm_extract-%j.out
#SBATCH --error=/data1/work/j0987341/aadya/research/DriveWeaver/logs/vlm_extract-%j.err
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

SCRIPT_VLM=scripts/cache_vlm_nuscenes.py
VLM_ARGS=(
  --slots_path /work/data/slots/nuscenes_slots_full.pkl
  --output_dir /work/data/vlm_cache/nuscenes/
  --nuscenes_root /data/nuScenes
  --model_name Qwen/Qwen3-VL-2B-Thinking
  --layers 6 12 18 24
  --max_frames 40
  --resolution 256
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
# VLM Feature Extraction
########################################

log_section "VLM Feature Extraction Started"

echo ""
echo "========================================"
echo "Extracting VLM Features (Qwen3-VL-2B-Thinking)"
echo "========================================"
echo "Output: /work/data/vlm_cache/nuscenes/"
echo "Expected scenes: 850 (700 train + 150 val)"
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

    PYTHONPATH=${WORKDIR} python '${SCRIPT_VLM}' ${VLM_ARGS[*]}
  "

if [ $? -ne 0 ]; then
    echo "ERROR: VLM extraction failed!"
    exit 1
fi

# Count extracted scenes
NUM_SCENES=$(docker exec "${CONTAINER}" bash -c "ls /work/data/vlm_cache/nuscenes/*.npz 2>/dev/null | wc -l")

echo ""
echo "========================================"
echo "VLM Feature Extraction Complete!"
echo "End: $(date)"
echo "========================================"
echo "Extracted scenes: ${NUM_SCENES}"
echo "Cache location: /work/data/vlm_cache/nuscenes/"
echo ""
echo "Next step: Submit ablation2 training job"
echo "  sbatch slurm/ablation2_cjepa_vlm_planner.sh"

EXIT_CODE=$?

trap - TERM INT EXIT
cleanup

exit "${EXIT_CODE}"
