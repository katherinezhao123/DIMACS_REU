#!/bin/bash
#SBATCH --job-name=full-finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --account=general
#SBATCH --output=logs/%j_finetune_out.txt
#SBATCH --error=logs/%j_finetune_err.txt

module load cuda/12.1.0
source /cache/home/kz498/miniconda3_new/bin/activate
conda activate torch118py310

export HF_TOKEN="hf_your_token"
mkdir -p logs output_full finetuned_full

# Monitor GPU memory
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
    --format=csv -l 5 > logs/${SLURM_JOB_ID}_fpft_memory.csv &
NVIDIA_PID=$!

START=$(date +%s)
python3 /cache/home/kz498/Research/Train/fpft_model.py
END=$(date +%s)

kill $NVIDIA_PID

echo "================================"
echo "Training time: $((END - START)) seconds"
echo "Peak GPU memory:"
awk -F',' 'NR>1 {print $2}' logs/${SLURM_JOB_ID}_fpft_memory.csv | sort -n | tail -1
echo "================================"