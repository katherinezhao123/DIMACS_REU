#!/bin/bash
#SBATCH --job-name=benchmark-lora-test4-qwen3
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=general
#SBATCH --requeue
#SBATCH --output=test4/out.txt
#SBATCH --error=test4/err.txt

module load cuda/12.1.0
source /cache/home/kz498/miniconda3_new/bin/activate
conda activate lora_benchmark
export HF_HOME=/scratch/kz498/.cache/huggingface


mkdir -p ./test4/samples 

nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
    --format=csv -l 5 > test4/lora_memory.csv &
NVIDIA_PID=$!

START=$(date +%s)

lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen3-8B-Base,peft=/cache/home/kz498/Research/Train/lora/qwen3/test4/model,tokenizer=Qwen/Qwen3-8B-Base \
    --tasks gsm8k \
    --batch_size 4 \
    --num_fewshot 8 \
    --seed $SLURM_JOB_ID \
    --output_path ./test4/samples \
    --log_samples \
    --device cuda 

END=$(date +%s)
kill $NVIDIA_PID

echo "================================"
echo "Benchmark time: $((END - START)) seconds"
echo "Peak GPU memory:"
awk -F',' 'NR>1 {print $2}' test4/lora_memory.csv | sort -n | tail -1
echo "================================"


