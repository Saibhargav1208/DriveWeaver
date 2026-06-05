#!/bin/bash
#SBATCH --job-name=abl2_train
#SBATCH --output=/data1/work/j0987341/aadya/research/DriveWeaver/logs/ablation2_train-%j.out
#SBATCH --error=/data1/work/j0987341/aadya/research/DriveWeaver/logs/ablation2_train-%j.err
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=72:00:00
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

# VLM cache already exists - skip extraction!
VLM_CACHE_DIR=/work/data/vlm_cache/nuscenes

# Stage 1: C-JEPA + VLM World Model
SCRIPT1=training/train_cjepa_vlm.py
CONFIG1=default

TRAIN_ARGS1=(
  --config-name "${CONFIG1}"
)

# Stage 2: Planner B (with VLM-enhanced world model)
SCRIPT2=training/train_planner_b.py
CONFIG2=default

TRAIN_ARGS2=(
  --config-name "${CONFIG2}"
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
# Pre-Flight Checks
########################################

log_section "Pre-flight checks"

echo "Checking VLM cache..."
VLM_COUNT=$(docker exec "${CONTAINER}" bash -c "ls ${VLM_CACHE_DIR}/*.npz 2>/dev/null | wc -l")
echo "VLM cache files: ${VLM_COUNT}"

if [ "${VLM_COUNT}" -lt 850 ]; then
    echo "ERROR: VLM cache incomplete (found ${VLM_COUNT}/850)"
    exit 1
fi

echo "✓ VLM cache complete (${VLM_COUNT} scenes)"

echo ""
echo "Checking slots data..."
docker exec "${CONTAINER}" bash -c "ls -lh /work/data/slots/nuscenes_slots_full.pkl"
echo "✓ Slots data ready"

echo ""
echo "Checking C-JEPA checkpoint (for world model initialization)..."
CJEPA_CKPT=$(docker exec "${CONTAINER}" bash -c "ls /work/checkpoints/cjepa/best_model.pth 2>/dev/null || echo 'NOT_FOUND'")
if [ "${CJEPA_CKPT}" == "NOT_FOUND" ]; then
    echo "WARNING: No C-JEPA checkpoint found - will train from scratch"
else
    echo "✓ C-JEPA checkpoint ready"
fi

########################################
# Ablation 2 Training Pipeline
########################################

log_section "Ablation 2: C-JEPA+VLM → Planner B"

echo ""
echo "========================================"
echo "STAGE 1/2: Training C-JEPA + VLM World Model"
echo "========================================"
echo "Duration: ~24-36 hours"
echo "Epochs: 50"
echo "Input: slots (850 scenes) + VLM cache (850 scenes)"
echo "Output: /work/checkpoints/cjepa_vlm/best_model.pth"
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
    echo 'Starting C-JEPA + VLM training...'
    echo ''

    PYTHONPATH=${WORKDIR} python '${SCRIPT1}' ${TRAIN_ARGS1[*]}
  "

STAGE1_EXIT=$?

if [ ${STAGE1_EXIT} -ne 0 ]; then
    echo "ERROR: C-JEPA + VLM training failed with exit code ${STAGE1_EXIT}"
    exit 1
fi

echo ""
echo "✓ Stage 1 complete!"
echo ""

echo ""
echo "========================================"
echo "STAGE 2/2: Training Planner B (VLM-enhanced)"
echo "========================================"
echo "Duration: ~12-24 hours"
echo "Epochs: 100"
echo "Input: C-JEPA+VLM world model + VLM cache"
echo "Output: /work/checkpoints/planner_b/best_model.pth"
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
    echo 'Starting Planner B training...'
    echo ''

    PYTHONPATH=${WORKDIR} python '${SCRIPT2}' ${TRAIN_ARGS2[*]}
  "

STAGE2_EXIT=$?

if [ ${STAGE2_EXIT} -ne 0 ]; then
    echo "ERROR: Planner B training failed with exit code ${STAGE2_EXIT}"
    exit 1
fi

echo ""
echo "✓ Stage 2 complete!"
echo ""

########################################
# Summary
########################################

log_section "Ablation 2 Training Complete!"

echo "Duration: $(date)"
echo ""
echo "Pipeline: videosaur slots → C-JEPA+VLM → Planner B"
echo ""
echo "Checkpoints saved:"
echo "  ✓ VLM cache:    ${VLM_CACHE_DIR}/ (850 scenes, 29GB)"
echo "  ✓ World model:  /work/checkpoints/cjepa_vlm/best_model.pth"
echo "  ✓ Planner:      /work/checkpoints/planner_b/best_model.pth"
echo ""
echo "Next steps:"
echo "  1. Evaluate world model: python evaluation/rollout.py --model cjepa_vlm"
echo "  2. Evaluate planner: python scripts/inference_planner_b.py"
echo "  3. Compare with Ablation 1 (baseline C-JEPA)"
echo ""

EXIT_CODE=0

trap - TERM INT EXIT
cleanup

exit "${EXIT_CODE}"
