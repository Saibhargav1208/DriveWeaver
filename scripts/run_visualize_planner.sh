#!/bin/bash
#
# Quick test script to visualize Ablation 1 planner predictions
#

CONTAINER=aadya_driveweaver

echo "Running Planner Visualization (Ablation 1)..."

docker exec ${CONTAINER} bash -c "
cd /work
PYTHONPATH=/work python scripts/visualize_planner.py \
    --world_model_ckpt /work/checkpoints/cjepa/best_model.pth \
    --planner_ckpt /work/checkpoints/planner_a/best_model.pth \
    --output_dir /work/outputs/vis_planner_a \
    --num_samples 50
"

echo ""
echo "✓ Done! Visualizations saved to:"
echo "  /work/outputs/vis_planner_a"
echo "  (Host: /data1/work/j0987341/aadya/research/DriveWeaver/outputs/vis_planner_a)"
