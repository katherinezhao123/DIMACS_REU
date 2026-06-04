#!/bin/bash
#SBATCH --job-name=gsm8k-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --account=general
#SBATCH --output=logs/Base/tests/%j_test.txt
#SBATCH --error=logs/Base/tests/%j_test.txt

module load cuda/12.1.0
source /cache/home/kz498/miniconda3_new/bin/activate
conda activate torch118py310

export HF_TOKEN="hf_your_token"

# Start GPU memory logging in background
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
    --format=csv -l 5 > logs/Base/tests/${SLURM_JOB_ID}_base_test_memory.csv &
NVIDIA_PID=$!

# Run benchmark with timing
START=$(date +%s)

lm_eval --model hf \
    --model_args pretrained=LiquidAI/LFM2.5-1.2B-Base \
    --tasks gsm8k \
    --num_fewshot 8 \
    --output_path ./eval_base/run_test_${SLURM_JOB_ID} \
    --seed 1 \
    --device cuda

END=$(date +%s)

# Stop memory logging
kill $NVIDIA_PID

# Print summary
echo "================================"
echo "Total time: $((END - START)) seconds"
echo "Peak GPU memory:"
awk -F',' 'NR>1 {print $2}' logs/Base/tests/${SLURM_JOB_ID}_base_test_memory.csv | sort -n | tail -1
echo "================================"