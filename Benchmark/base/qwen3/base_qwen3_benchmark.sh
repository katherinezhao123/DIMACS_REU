#!/bin/bash
#SBATCH --job-name=qwen3-baseline-benchmark
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=general
#SBATCH --output=out.txt
#SBATCH --error=err.txt

module load cuda/12.1.0
source /cache/home/kz498/miniconda3_new/bin/activate
conda activate lora_benchmark
export HF_HOME=/scratch/kz498/huggingface
export HF_DATASETS_CACHE=/scratch/kz498/huggingface/datasets 
export TRANSFORMERS_CACHE=/scratch/kz498/huggingface/hub

# Start GPU memory logging in background
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
    --format=csv -l 5 > memory.csv &
NVIDIA_PID=$!

# Run benchmark with timing
START=$(date +%s)

lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen3-8B-Base \
    --tasks gsm8k \
    --num_fewshot 8 \
    --batch_size 4 \
    --output_path ./model \
    --seed $SLURM_JOB_ID \
    --device cuda \
    

END=$(date +%s)

# Stop memory logging
kill $NVIDIA_PID

# Print summary
echo "================================"
echo "Total time: $((END - START)) seconds"
echo "Peak GPU memory:"
awk -F',' 'NR>1 {print $2}' memory.csv | sort -n | tail -1
echo "================================"