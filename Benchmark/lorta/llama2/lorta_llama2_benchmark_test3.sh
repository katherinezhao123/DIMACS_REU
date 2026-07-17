#!/bin/bash
#SBATCH --job-name=benchmark-lorta-test3-llama2
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=general
#SBATCH --requeue
#SBATCH --output=test3/out.txt
#SBATCH --error=test3/err.txt

module load cuda/12.1.0
source /cache/home/kz498/miniconda3_new/bin/activate
conda activate /scratch/kz498/conda_envs/lorta_benchmark
export HF_HOME=/scratch/kz498/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p ./test3/samples 

nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
    --format=csv -l 5 > test3/lora_memory.csv &
NVIDIA_PID=$!

START=$(date +%s)

lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=/cache/home/kz498/Research/Train/lorta/llama2/test3/model,tokenizer=meta-llama/Llama-2-7b-hf \
    --tasks gsm8k \
    --batch_size 1 \
    --num_fewshot 8 \
    --seed $SLURM_JOB_ID \
    --output_path ./test3/samples \
    --log_samples \
    --device cuda 

END=$(date +%s)
kill $NVIDIA_PID

echo "================================"
echo "Benchmark time: $((END - START)) seconds"
echo "Peak GPU memory:"
awk -F',' 'NR>1 {print $2}' test3/lora_memory.csv | sort -n | tail -1
echo "================================"


