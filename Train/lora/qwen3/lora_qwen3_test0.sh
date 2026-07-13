#!/bin/bash
#SBATCH --job-name=lora-qwen3-test0
#SBATCH --partition=gpu-redhat            # Amarel's GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --cpus-per-task=8          # CPU cores for data loading
#SBATCH --gres=gpu:2               # request 1 GPU
#SBATCH --mem=32G                  # RAM (not VRAM — system memory)
#SBATCH --time=9:00:00            # 12 hours, adjust based on your estimate
#SBATCH --output=test0/lora_out.txt   # stdout — %j is the job ID
#SBATCH --error=test0/lora_err.txt    # stderr separately so errors are easy to find

module load cuda/12.1.0
source /cache/home/kz498/miniconda3_new/bin/activate
conda activate lora_train
export HF_HOME=/scratch/kz498/huggingface
export HF_DATASETS_CACHE=/scratch/kz498/huggingface/datasets 
export TRANSFORMERS_CACHE=/scratch/kz498/huggingface/hub

mkdir -p ./test0/model ./test0/checkpoints

# Monitor GPU memory
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
    --format=csv -l 5 > test0/lora_memory.csv &
NVIDIA_PID=$!

START=$(date +%s)
python3 /cache/home/kz498/Research/Train/lora/qwen3/lora_qwen3.py \
        --r 8 \
        --lora_alpha 16 \
        --lr 2e-5 \
        --num_train_epochs 1 \
        --per_device 2 \
        --gradient_accum 4 \
        --lora_dropout 0.05 \
        --warmup_ratio 0.03 \
        --lr_scheduler linear \
        --output_dir ./test0/checkpoints \
        --save_dir ./test0/model \
        --target_mod q_proj k_proj v_proj o_proj gate_proj up_proj down_proj

END=$(date +%s)


kill $NVIDIA_PID

echo "================================"
echo "Training time: $((END - START)) seconds"
echo "Peak GPU memory:"
awk -F',' 'NR>1 {print $2}' test0/lora_memory.csv | sort -n | tail -1
echo "================================"