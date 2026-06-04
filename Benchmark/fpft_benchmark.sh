#!/bin/bash
#SBATCH --job-name=benchmark-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=general
#SBATCH --output=logs/%j_benchmark_fpft_out.txt
#SBATCH --error=logs/%j_benchmark_fpft_err.txt

module load cuda/12.1.0
source /cache/home/kz498/miniconda3_new/bin/activate
conda activate torch118py310

export HF_TOKEN="hf_your_token"
mkdir -p logs eval_fpft

nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
    --format=csv -l 5 > logs/${SLURM_JOB_ID}_fpft_memory.csv &
NVIDIA_PID=$!

START=$(date +%s)

lm_eval --model hf \
    --model_args pretrained=/cache/home/kz498/Research/Train/fpft\
    --tasks gsm8k \
    --num_fewshot 8 \
    --seed $SLURM_JOB_ID \
    --output_path ./eval_fpft \
    --device cuda

END=$(date +%s)
kill $NVIDIA_PID

echo "================================"
echo "Benchmark time: $((END - START)) seconds"
echo "Peak GPU memory:"
awk -F',' 'NR>1 {print $2}' logs/${SLURM_JOB_ID}_fpft_memory.csv | sort -n | tail -1
echo "================================"