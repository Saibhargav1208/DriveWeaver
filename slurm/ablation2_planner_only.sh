#!/bin/bash
#SBATCH --job-name=abl2_plan
#SBATCH --output=/data1/work/j0987341/aadya/research/DriveWeaver/logs/ablation2_planner-%j.out
#SBATCH --error=/data1/work/j0987341/aadya/research/DriveWeaver/logs/ablation2_planner-%j.err
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

SCRIPT=training/train_planner_b.py
CONFIG=default

VLM_CACHE_DIR=/work/data/vlm_cache/nuscenes
SLOTS_PATH=/work/data/slots/nuscenes_slots_full.pkl
CJEPA_VLM_CKPT=/work/checkpoints/cjepa_vlm/best_model.pth
PLANNER_DIR=/work/checkpoints/planner_b

# Keep the full ablation setting by default. Override at submit time with:
#   sbatch --export=ALL,MAX_EPOCHS=20 slurm/ablation2_planner_only.sh
MAX_EPOCHS="${MAX_EPOCHS:-100}"
RESET_PLANNER_DIR="${RESET_PLANNER_DIR:-true}"

TRAIN_ARGS=(
  --config-name "${CONFIG}"
  data.slots_path="${SLOTS_PATH}"
  vlm_guidance.cache_dir="${VLM_CACHE_DIR}/"
  checkpoints.cjepa="${CJEPA_VLM_CKPT}"
  checkpoints.planner=null
  training.checkpoint_dir="${PLANNER_DIR}"
  training.max_epochs="${MAX_EPOCHS}"
)

RESTART_CONTAINER_ON_EXIT="false"

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
echo "Max epochs: ${MAX_EPOCHS}"

########################################
# GPU Info
########################################

log_section "GPU status"

nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv
VISIBLE_UUIDS="$(get_visible_gpu_uuids)"
echo "NVIDIA_VISIBLE_DEVICES=${VISIBLE_UUIDS}"

########################################
# Container and Data Checks
########################################

log_section "Container check"

docker ps | grep "${CONTAINER}"
docker exec "${CONTAINER}" pwd

log_section "CUDA preflight inside container"

docker exec \
  -e NVIDIA_VISIBLE_DEVICES="${VISIBLE_UUIDS}" \
  -e CUDA_VISIBLE_DEVICES="${VISIBLE_UUIDS}" \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES=${VISIBLE_UUIDS}
    nvidia-smi
    python -c 'import torch; print(\"torch\", torch.__version__); print(\"cuda_available\", torch.cuda.is_available()); print(\"device_count\", torch.cuda.device_count()); assert torch.cuda.is_available(), \"CUDA is not available inside container\"'
  "

log_section "Pre-flight checks"

echo "Checking slots data..."
docker exec "${CONTAINER}" bash -c "ls -lh '${SLOTS_PATH}'"
echo "Slots data ready"

echo ""
echo "Checking VLM cache..."
VLM_COUNT=$(docker exec "${CONTAINER}" bash -c "ls '${VLM_CACHE_DIR}'/*.npz 2>/dev/null | wc -l")
echo "VLM cache files: ${VLM_COUNT}"
if [ "${VLM_COUNT}" -lt 850 ]; then
  echo "ERROR: VLM cache incomplete (found ${VLM_COUNT}/850)"
  exit 1
fi
echo "VLM cache complete"

echo ""
echo "Checking C-JEPA + VLM checkpoint..."
docker exec "${CONTAINER}" bash -c "ls -lh '${CJEPA_VLM_CKPT}'"
echo "C-JEPA + VLM checkpoint ready"

if [[ "${RESET_PLANNER_DIR}" == "true" ]]; then
  log_section "Reset planner checkpoint directory"
  BACKUP_DIR="${PLANNER_DIR}_before_${SLURM_JOB_ID}_$(date +%Y%m%d-%H%M%S)"
  docker exec "${CONTAINER}" bash -c "
    set -euo pipefail
    if [ -d '${PLANNER_DIR}' ]; then
      echo 'Archiving existing planner dir to ${BACKUP_DIR}'
      mv '${PLANNER_DIR}' '${BACKUP_DIR}'
    fi
    mkdir -p '${PLANNER_DIR}'
  "
fi

########################################
# Planner Training
########################################

log_section "Ablation 2 Planner B from scratch"

echo "Input world model: ${CJEPA_VLM_CKPT}"
echo "Planner output: ${PLANNER_DIR}/best_model.pth"
echo "Command: PYTHONPATH=${WORKDIR} python ${SCRIPT} ${TRAIN_ARGS[*]}"

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

    PYTHONPATH=${WORKDIR} python '${SCRIPT}' ${TRAIN_ARGS[*]}
  "

echo ""
echo "========================================"
echo "Ablation 2 Planner B Complete!"
echo "End: $(date)"
echo "========================================"
echo "Checkpoints:"
echo "  World model: ${CJEPA_VLM_CKPT}"
echo "  Planner:     ${PLANNER_DIR}/best_model.pth"

EXIT_CODE=0

trap - TERM INT EXIT
cleanup

exit "${EXIT_CODE}"
