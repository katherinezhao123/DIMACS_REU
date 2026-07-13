#!/bin/bash
#SBATCH --job-name=fpft-liquidai-test0
#SBATCH --partition=gpu-redhat            # Amarel's GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --cpus-per-task=8          # CPU cores for data loading
#SBATCH --gres=gpu:4               # request 1 GPU
#SBATCH --mem=32G                  # RAM (not VRAM — system memory)
#SBATCH --time=9:00:00            # 12 hours, adjust based on your estimate
#SBATCH --output=test0/out.txt   # stdout — %j is the job ID
#SBATCH --error=test0/err.txt    # stderr separately so errors are easy to find

module load cuda/12.1.0
source /cache/home/kz498/miniconda3_new/bin/activate
conda activate lora_train
export HF_HOME=/cache/home/kz498/.cache/huggingface

mkdir -p ./test0/model ./test0/checkpoints

# Monitor GPU memory
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
    --format=csv -l 5 > test0/lora_memory.csv &
NVIDIA_PID=$!

START=$(date +%s)
python3 /cache/home/kz498/Research/Train/fpft/liquidai/fpft_liquidai.py \
        --lr 5e-5 \
        --num_train_epochs 4 \
        --per_device 4 \
        --gradient_accum 4 \
        --warmup_ratio 0.06 \
        --lr_scheduler cosine \
        --output_dir ./test0/checkpoints \
        --save_dir ./test0/model

END=$(date +%s)


kill $NVIDIA_PID

echo "================================"
echo "Training time: $((END - START)) seconds"
echo "Peak GPU memory:"
awk -F',' 'NR>1 {print $2}' test0/lora_memory.csv | sort -n | tail -1
echo "================================"