#!/bin/bash
#SBATCH --job-name=lorta-finetune-test1
#SBATCH --partition=gpu-redhat            # Amarel's GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --cpus-per-task=8          # CPU cores for data loading
#SBATCH --gres=gpu:1               # request 1 GPU
#SBATCH --mem=32G                  # RAM (not VRAM — system memory)
#SBATCH --time=9:00:00            # 12 hours, adjust based on your estimate
#SBATCH --output=test1/lorta_out.txt   # stdout — %j is the job ID
#SBATCH --error=test1/lorta_err.txt    # stderr separately so errors are easy to find

module load cuda/12.1.0
source /cache/home/kz498/miniconda3_new/bin/activate
conda activate lorta_train
export HF_HOME=/cache/home/kz498/.cache/huggingface


mkdir -p ./test1/model ./test1/checkpoints

# Monitor GPU memory
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
    --format=csv -l 5 > test1/lorta_memory.csv &
NVIDIA_PID=$!

START=$(date +%s)
python3 /cache/home/kz498/Research/Train/lorta/lorta_model.py \
        --r 4 \
        --lorta_alpha 8 \
        --lorta_dropout 0.05 \
        --per_device 16 \
        --gradient_accum 4 \
        --num_train_epochs 3 \
        --lr 2e-4 \
        --warmup_ratio 0.06 \
        --lr_scheduler cosine \
        --output_dir ./test1/checkpoints \
        --save_dir ./test1/model 

END=$(date +%s)


kill $NVIDIA_PID

echo "================================"
echo "Training time: $((END - START)) seconds"
echo "Peak GPU memory:"
awk -F',' 'NR>1 {print $2}' test1/lorta_memory.csv | sort -n | tail -1
echo "================================"