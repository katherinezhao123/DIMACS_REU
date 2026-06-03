#!/bin/bash
#SBATCH --job-name=lora-finetune
#SBATCH --partition=gpu            # Amarel's GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # CPU cores for data loading
#SBATCH --gres=gpu:1               # request 1 GPU
#SBATCH --mem=64G                  # RAM (not VRAM — system memory)
#SBATCH --time=12:00:00            # 12 hours, adjust based on your estimate
#SBATCH --output=logs/%j_out.txt   # stdout — %j is the job ID
#SBATCH --error=logs/%j_err.txt    # stderr separately so errors are easy to find
#SBATCH --mail-type=BEGIN,END,FAIL # email on job events
#SBATCH --mail-user=YOURNETID@rutgers.edu

# Load the right modules
module purge
module load cuda 



# # Create a fresh venv on the cluster
# python3 -m venv .venv
source .venv/bin/activate

export HF_TOKEN=""

pip install 
# Make log dirs so #SBATCH --output doesn't fail
mkdir -p logs

# --- Run ---
python3 modeltraining.py