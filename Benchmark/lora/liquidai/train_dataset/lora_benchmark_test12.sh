#!/bin/bash
#SBATCH --job-name=benchmark-lora-test12-train
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=general
#SBATCH --output=test12/lora_out.txt
#SBATCH --error=test12/lora_err.txt

module load cuda/12.1.0
source /cache/home/kz498/miniconda3_new/bin/activate
conda activate lora_benchmark
export HF_HOME=/cache/home/kz498/.cache/huggingface


mkdir -p ./test12/samples 

nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
    --format=csv -l 5 > test12/lora_memory.csv &
NVIDIA_PID=$!

START=$(date +%s)

lm_eval --model hf \
    --model_args pretrained=LiquidAI/LFM2.5-8B-A1B-Base,peft=/cache/home/kz498/Research/Train/lora/test12/model,tokenizer=LiquidAI/LFM2.5-8B-A1B-Base \
    --tasks gsm8k_train \
    --include_path /cache/home/kz498/gsm8k_train_task \
    --batch_size 4 \
    --num_fewshot 8 \
    --seed $SLURM_JOB_ID \
    --output_path ./test12/samples \
    --log_samples \
    --device cuda 

END=$(date +%s)
kill $NVIDIA_PID

echo "================================"
echo "Benchmark time: $((END - START)) seconds"
echo "Peak GPU memory:"
awk -F',' 'NR>1 {print $2}' test12/lora_memory.csv | sort -n | tail -1
echo "================================"

