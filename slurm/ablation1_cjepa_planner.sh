#!/bin/bash
#SBATCH --job-name=abl1_cjepa
#SBATCH --output=/data1/work/j0987341/aadya/research/DriveWeaver/logs/ablation1-%j.out
#SBATCH --error=/data1/work/j0987341/aadya/research/DriveWeaver/logs/ablation1-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --partition=h100
#SBATCH --signal=B:TERM@5

set -euo pipefail
set -x

########################################
# User Settings
########################################

CONTAINER="aadya_driveweaver"

WORKDIR=/work

SCRIPT1=training/train_cjepa.py
CONFIG1=default

TRAIN_ARGS1=(
  --config-name "${CONFIG1}"
)

SCRIPT2=training/train_planner_a.py
CONFIG2=default

TRAIN_ARGS2=(
  --config-name "${CONFIG2}"
)
########################################
# python '${SCRIPT}' ${TRAIN_ARGS[*]}
########################################

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
# Training
########################################

log_section "Training started"

echo ""
echo "========================================"
echo "Step 1: Training C-JEPA World Model"
echo "========================================"

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

    PYTHONPATH=${WORKDIR} python '${SCRIPT1}' ${TRAIN_ARGS1[*]}
  "
if [ $? -ne 0 ]; then
    echo "ERROR: C-JEPA training failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "Step 2: Training Planner (Ablation 1)"
echo "========================================"

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

    PYTHONPATH=${WORKDIR} python '${SCRIPT2}' ${TRAIN_ARGS2[*]}
  "
if [ $? -ne 0 ]; then
    echo "ERROR: Planner training failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "Ablation 1 Complete!"
echo "End: $(date)"
echo "========================================"
echo "Checkpoints:"
echo "  World model: /work/checkpoints/cjepa/best_model.pth"
echo "  Planner:     /work/checkpoints/planner_a/best_model.pth"

EXIT_CODE=$?

trap - TERM INT EXIT
cleanup

exit "${EXIT_CODE}"