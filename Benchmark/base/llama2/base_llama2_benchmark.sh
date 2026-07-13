#!/bin/bash
#SBATCH --job-name=gsm8k-baseline-base
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --account=general
#SBATCH --output=base/out.txt
#SBATCH --error=base/err.txt

module load cuda/12.1.0
source /cache/home/kz498/miniconda3_new/bin/activate
conda activate lora_benchmark


# Start GPU memory logging in background
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
    --format=csv -l 5 > base/memory.csv &
NVIDIA_PID=$!

# Run benchmark with timing
START=$(date +%s)

lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks gsm8k \
    --num_fewshot 8 \
    --output_path ./base \
    --seed $SLURM_JOB_ID \
    --batch_size auto \
    --device cuda \


END=$(date +%s)

# Stop memory logging
kill $NVIDIA_PID

# Print summary
echo "================================"
echo "Total time: $((END - START)) seconds"
echo "Peak GPU memory:"
awk -F',' 'NR>1 {print $2}' base/memory.csv | sort -n | tail -1
echo "================================"