#!/bin/bash
#SBATCH --job-name=lorta-padded-llama2-test8
#SBATCH --partition=gpu-redhat            # Amarel's GPU partition
#SBATCH --nodes=1
#SBATCH --exclude=gpu017
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --cpus-per-task=8          # CPU cores for data loading
#SBATCH --gres=gpu:1               # request 1 GPU
#SBATCH --mem=32G                  # RAM (not VRAM — system memory)
#SBATCH --time=9:00:00            # 12 hours, adjust based on your estimate
#SBATCH --output=test8/out.txt   # stdout — %j is the job ID
#SBATCH --error=test8/err.txt    # stderr separately so errors are easy to find

module load cuda/12.1.0
source /cache/home/kz498/miniconda3_new/bin/activate
conda activate lorta_train
export HF_HOME=/scratch/kz498/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/kz498/huggingface/datasets 
export TRANSFORMERS_CACHE=/scratch/kz498/huggingface/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p ./test8/model ./test8/checkpoints

# Monitor GPU memory
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
    --format=csv -l 5 > test8/memory.csv &
NVIDIA_PID=$!

START=$(date +%s)
python3 /cache/home/kz498/Research/Train/lorta_padded/llama2/lorta_padded_llama2.py \
        --r 128 \
        --lorta_alpha 4 \
        --lorta_dropout 0.05 \
        --per_device 2 \
        --gradient_accum 32 \
        --num_train_epochs 6 \
        --lr 5e-3 \
        --warmup_ratio 0.03 \
        --lr_scheduler cosine \
        --output_dir ./test8/checkpoints \
        --save_dir ./test8/model \
        --target_mod q_proj v_proj k_proj o_proj

END=$(date +%s)


kill $NVIDIA_PID

echo "================================"
echo "Training time: $((END - START)) seconds"
echo "Peak GPU memory:"
awk -F',' 'NR>1 {print $2}' test8/memory.csv | sort -n | tail -1
echo "================================"