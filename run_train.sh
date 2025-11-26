#!/bin/bash
#SBATCH --job-name=layoutlm_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

# 1. Clean Environment
module purge
module load python/3.13.5
module load cuda/12.1.1

# 2. Activate Venv (CRITICAL: Do this BEFORE running python commands)
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
# 3. Debugging Info
echo "--- DEBUG INFO ---"
echo "Hostname: $(hostname)"
echo "Python:"
which python
python --version
echo "Checking PyTorch..."
python -c "import torch; print(f'Torch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
echo "------------------"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 4. Run Training
export PYTHONPATH=$PYTHONPATH:.
python -m src.train