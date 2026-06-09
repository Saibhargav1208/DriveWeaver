#!/bin/bash
#SBATCH --job-name=abl2_pscore
#SBATCH --output=/data1/work/j0987341/aadya/research/DriveWeaver/logs/ablation2_planner_scoreloss-%j.out
#SBATCH --error=/data1/work/j0987341/aadya/research/DriveWeaver/logs/ablation2_planner_scoreloss-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --partition=h100
#SBATCH --signal=B:TERM@5

set -euo pipefail
set -x

CONTAINER=aadya_driveweaver
WORKDIR=/work
SCRIPT=training/train_planner_b.py
CONFIG=default

CACHE_DIR=/work/data/future_slots/cjepa_vlm
PLANNER_DIR=/work/checkpoints/planner_b
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
RESET_PLANNER_DIR="${RESET_PLANNER_DIR:-true}"

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
  docker ps | grep "${CONTAINER}" || true
  exit "${exit_code}"
}

get_visible_gpu_uuids() {
  nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -
}

trap cleanup TERM INT EXIT

log_section "Job started"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Cache: ${CACHE_DIR}"
echo "Planner output: ${PLANNER_DIR}"
echo "Batch size: ${BATCH_SIZE}"
echo "Max epochs: ${MAX_EPOCHS}"

log_section "GPU status"
nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv
VISIBLE_UUIDS="$(get_visible_gpu_uuids)"
echo "NVIDIA_VISIBLE_DEVICES=${VISIBLE_UUIDS}"

log_section "Container/CUDA preflight"
docker ps | grep "${CONTAINER}"
docker exec \
  -e NVIDIA_VISIBLE_DEVICES="${VISIBLE_UUIDS}" \
  -e CUDA_VISIBLE_DEVICES="${VISIBLE_UUIDS}" \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES=${VISIBLE_UUIDS}
    cd '${WORKDIR}'
    nvidia-smi
    python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); assert torch.cuda.is_available()'
    ls -lh '${CACHE_DIR}/train.pt' '${CACHE_DIR}/val.pt'
  "

if [[ "${RESET_PLANNER_DIR}" == "true" ]]; then
  log_section "Reset planner checkpoint directory"
  BACKUP_DIR="${PLANNER_DIR}_before_scoreloss_${SLURM_JOB_ID}_$(date +%Y%m%d-%H%M%S)"
  docker exec "${CONTAINER}" bash -c "
    set -euo pipefail
    if [ -d '${PLANNER_DIR}' ]; then
      echo 'Archiving existing planner dir to ${BACKUP_DIR}'
      mv '${PLANNER_DIR}' '${BACKUP_DIR}'
    fi
    mkdir -p '${PLANNER_DIR}'
  "
fi

log_section "Train Planner B with proposal score loss"
docker exec \
  -e NVIDIA_VISIBLE_DEVICES="${VISIBLE_UUIDS}" \
  -e CUDA_VISIBLE_DEVICES="${VISIBLE_UUIDS}" \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${WORKDIR}'
    export CUDA_VISIBLE_DEVICES=${VISIBLE_UUIDS}
    PYTHONPATH=${WORKDIR} python '${SCRIPT}' \
      --config-name '${CONFIG}' \
      ++data.future_slots_cache_dir='${CACHE_DIR}' \
      data.batch_size='${BATCH_SIZE}' \
      data.num_workers='${NUM_WORKERS}' \
      training.checkpoint_dir='${PLANNER_DIR}' \
      training.max_epochs='${MAX_EPOCHS}' \
      checkpoints.planner=null \
      loss.lambda_score_ce=1.0 \
      loss.lambda_score_kl=0.0 \
      loss.score_temperature=0.5 \
      loss.lambda_all_ade=0.05 \
      loss.lambda_diversity=0.02
  "

log_section "Score-loss Planner B training complete"
docker exec "${CONTAINER}" bash -c "ls -lh '${PLANNER_DIR}'"

trap - TERM INT EXIT
cleanup
