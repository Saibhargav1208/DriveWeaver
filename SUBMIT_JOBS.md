# How to Submit SLURM Jobs - Ablations 1 & 2

**Date:** 2026-05-27  
**Status:** ✅ Ready to submit (optimizations applied)

---

## ✅ Pre-Submission Checklist

All optimizations have been applied:

- [x] **Ablation 1 (C-JEPA):** `num_workers: 12` in `configs/cjepa/default.yaml`
- [x] **Ablation 2 (C-JEPA+VLM):** `num_workers: 12` in `configs/cjepa_vlm/default.yaml`
- [x] **Dataloader:** Added `persistent_workers=True` and `prefetch_factor=4`
- [x] **SLURM scripts:** Already configured with `--gres=gpu:1` (single GPU)

---

## 🚀 Submit Jobs

### Option 1: Submit Both Jobs Sequentially

```bash
# Navigate to DriveWeaver directory
cd /data1/work/j0987341/aadya/research/DriveWeaver

# Submit Ablation 1 (C-JEPA → Planner)
sbatch slurm/ablation1_cjepa_planner.sh

# Submit Ablation 2 (C-JEPA + VLM → Planner)
sbatch slurm/ablation2_cjepa_vlm_planner.sh
```

Both will run in parallel if resources are available.

### Option 2: Submit with Dependency (Ablation 2 waits for Ablation 1)

```bash
# Submit Ablation 1 first
JOB1=$(sbatch --parsable slurm/ablation1_cjepa_planner.sh)
echo "Submitted Ablation 1: Job ID $JOB1"

# Submit Ablation 2 to start after Ablation 1 completes
JOB2=$(sbatch --dependency=afterok:$JOB1 --parsable slurm/ablation2_cjepa_vlm_planner.sh)
echo "Submitted Ablation 2: Job ID $JOB2 (waits for $JOB1)"
```

### Option 3: Submit Just One Job

```bash
# Just Ablation 1
sbatch slurm/ablation1_cjepa_planner.sh

# OR just Ablation 2
sbatch slurm/ablation2_cjepa_vlm_planner.sh
```

---

## 📊 Job Specifications

### Ablation 1: C-JEPA → Planner

```
Job Name:    abl1_cjepa
Partition:   h100
GPU:         1x H100 (80GB)
Memory:      64GB CPU RAM
Time Limit:  12 hours
```

**Pipeline:**
1. Train C-JEPA world model (~3-5 hours)
2. Train Planner head on frozen C-JEPA

**Expected GPU Utilization:** 30-50%

**Outputs:**
- Checkpoints: `/work/checkpoints/cjepa/best_model.pth`
- Planner: `/work/checkpoints/planner_a/best_model.pth`
- Logs: `logs/ablation1-<jobid>.{out,err}`

---

### Ablation 2: C-JEPA + VLM → Planner

```
Job Name:    abl2_vlm
Partition:   h100
GPU:         1x H100 (80GB)
Memory:      64GB CPU RAM
Time Limit:  24 hours
```

**Pipeline:**
1. Extract VLM features from nuScenes images (~1-2 hours)
2. Train C-JEPA + VLM world model (~4-6 hours)
3. Train Planner head on frozen C-JEPA+VLM

**Expected GPU Utilization:** 30-50%

**Outputs:**
- VLM cache: `/work/data/vlm_cache/nuscenes/`
- Checkpoints: `/work/checkpoints/cjepa_vlm/best_model.pth`
- Planner: `/work/checkpoints/planner_b/best_model.pth`
- Logs: `logs/ablation2-<jobid>.{out,err}`

---

## 📈 Monitor Jobs

### Check Job Status

```bash
# List your jobs
squeue -u $USER

# Detailed status
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Check specific job
squeue -j <job_id>
```

### Monitor GPU Utilization

```bash
# SSH to the node running your job (find node with squeue)
ssh <node_name>

# Watch GPU usage (should see 30-50% utilization)
watch -n 1 nvidia-smi

# Or more detailed monitoring
nvidia-smi dmon -s uct -d 1
```

### Tail Logs (Real-time)

```bash
# For ablation 1
tail -f logs/ablation1-<jobid>.out
tail -f logs/ablation1-<jobid>.err

# For ablation 2
tail -f logs/ablation2-<jobid>.out
tail -f logs/ablation2-<jobid>.err
```

### Check Training Progress

```bash
# See latest checkpoint
ls -lth /data1/work/j0987341/aadya/research/DriveWeaver/checkpoints/cjepa/

# Inside docker (check from host)
docker exec aadya_driveweaver ls -lth /work/checkpoints/cjepa/
```

---

## ⚠️ Troubleshooting

### If Job Gets Killed Again

1. **Check logs:**
   ```bash
   tail -100 logs/ablation1-<jobid>.err
   ```

2. **Verify GPU utilization was good:**
   ```bash
   grep -i "gpu\|utilization" logs/ablation1-<jobid>.out
   ```

3. **If still <10% GPU util:**
   - Increase `num_workers` to 16 in config
   - Increase `prefetch_factor` to 8 in `datasets/slot_dataset.py`

### If Out of Memory (Unlikely)

Current memory usage is ~1GB / 80GB, so OOM is very unlikely. But if it happens:

```yaml
# Reduce batch size in config
data:
  batch_size: 4  # or 8
```

### If Training is Slow

Expected speeds:
- **Ablation 1:** ~3-5 hours for C-JEPA training
- **Ablation 2:** ~4-6 hours for C-JEPA+VLM training

If slower:
- Check `nvidia-smi` - should see 30-50% GPU util
- Check `htop` - should see 12 python workers active
- Increase `num_workers` to 16

---

## 🎯 Success Criteria

Your jobs should:
- ✅ Complete without being killed by SLURM
- ✅ Show 30-50% GPU utilization
- ✅ Train in ~3-5 hours (Ablation 1) or ~5-8 hours (Ablation 2)
- ✅ Save checkpoints every 5 epochs
- ✅ Generate best_model.pth for each stage

---

## 📝 Quick Reference

```bash
# Submit both jobs
cd /data1/work/j0987341/aadya/research/DriveWeaver
sbatch slurm/ablation1_cjepa_planner.sh
sbatch slurm/ablation2_cjepa_vlm_planner.sh

# Check status
squeue -u $USER

# Cancel a job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER

# View job details
scontrol show job <job_id>

# View past job info
sacct -j <job_id> --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS,MaxVMSize
```

---

## 📞 Need Help?

If issues persist:
1. Check [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) for detailed troubleshooting
2. Share the error logs from `logs/ablation*-<jobid>.err`
3. Check GPU utilization in the output logs

Good luck! 🚀
