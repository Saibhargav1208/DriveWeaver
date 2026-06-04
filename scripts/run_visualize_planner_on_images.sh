#!/bin/bash
#
# Visualize planner predictions overlayed on nuScenes camera images
#

CONTAINER=aadya_driveweaver

echo "Running Planner Visualization on Images (Ablation 1)..."

docker exec ${CONTAINER} bash -c "
cd /work
PYTHONPATH=/work python scripts/visualize_planner_on_images.py \
    --world_model_ckpt /work/checkpoints/cjepa/best_model.pth \
    --planner_ckpt /work/checkpoints/planner_a/best_model.pth \
    --output_dir /work/outputs/vis_planner_a_images \
    --nuscenes_root /data/nuScenes \
    --num_samples 50
"

echo ""
echo "✓ Done! Image visualizations saved to:"
echo "  /work/outputs/vis_planner_a_images"
echo "  (Host: /data1/work/j0987341/aadya/research/DriveWeaver/outputs/vis_planner_a_images)"
