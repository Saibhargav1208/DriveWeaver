#!/bin/bash
#SBATCH --job-name=abl1_cache
#SBATCH --output=/data1/work/j0987341/aadya/research/DriveWeaver/logs/ablation1_precompute-%j.out
#SBATCH --error=/data1/work/j0987341/aadya/research/DriveWeaver/logs/ablation1_precompute-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --partition=h100
#SBATCH --signal=B:TERM@5

set -euo pipefail
set -x

CONTAINER=aadya_driveweaver
WORKDIR=/work
SCRIPT=scripts/precompute_cjepa_vlm_futures.py

SLOTS_PATH=/work/data/slots/nuscenes_slots_full.pkl
WORLD_MODEL_CKPT=/work/checkpoints/cjepa/best_model.pth
OUTPUT_DIR=/work/data/future_slots/cjepa
BATCH_SIZE="${BATCH_SIZE:-64}"
SAVE_DTYPE="${SAVE_DTYPE:-fp16}"
OVERWRITE_CACHE="${OVERWRITE_CACHE:-false}"

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
echo "Container: ${CONTAINER}"
echo "Output: ${OUTPUT_DIR}"
echo "Batch size: ${BATCH_SIZE}"

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
    ls -lh '${SLOTS_PATH}' '${WORLD_MODEL_CKPT}'
  "

log_section "Precompute C-JEPA-only future slots"
OVERWRITE_ARG=""
if [[ "${OVERWRITE_CACHE}" == "true" ]]; then
  OVERWRITE_ARG="--overwrite"
fi

docker exec \
  -e NVIDIA_VISIBLE_DEVICES="${VISIBLE_UUIDS}" \
  -e CUDA_VISIBLE_DEVICES="${VISIBLE_UUIDS}" \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${WORKDIR}'
    export CUDA_VISIBLE_DEVICES=${VISIBLE_UUIDS}
    PYTHONPATH=${WORKDIR} python '${SCRIPT}' \
      --slots_path '${SLOTS_PATH}' \
      --world_model_ckpt '${WORLD_MODEL_CKPT}' \
      --output_dir '${OUTPUT_DIR}' \
      --splits train val \
      --batch_size '${BATCH_SIZE}' \
      --save_dtype '${SAVE_DTYPE}' \
      --no_vlm_guidance \
      ${OVERWRITE_ARG}
  "

log_section "Precompute complete"
docker exec "${CONTAINER}" bash -c "ls -lh '${OUTPUT_DIR}'"

trap - TERM INT EXIT
cleanup
