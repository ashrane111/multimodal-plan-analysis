#!/bin/bash
#SBATCH --job-name=layoutlm_train      # Job name
#SBATCH --output=logs/train_%j.out     # Output log file (%j = job ID)
#SBATCH --error=logs/train_%j.err      # Error log file
#SBATCH --partition=sharing                # Partition name
#SBATCH --gres=gpu:a100:1              # Request 1 V100 GPU (Critical for your PyTorch version)
#SBATCH --nodes=1                      # Run on a single node
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --cpus-per-task=8              # Request 8 CPU cores for data loading
#SBATCH --mem=32G                      # Request 32GB RAM
#SBATCH --time=01:00:00                # Time limit (4 hours)

echo "Job started on $(hostname) at $(date)"

# DEBUG: Print what is currently loaded
echo "Modules before cleaning:"
module list

# AGGRESSIVE CLEANING
module unload cuda       # Try to unload generic cuda
module unload cuda/11.0  # Try to unload common defaults
module unload cuda/11.8
module unload gcc        # GCC often binds to CUDA

# Load fresh
module load cuda/12.1.1

# DEBUG: Print what we actually got
echo "Modules after loading:"
module list

# Verify CUDA is visible
nvidia-smi

# 2. Activate your virtual environment
source .venv/bin/activate

# 3. Set Python Path so it finds 'src'
export PYTHONPATH=$PYTHONPATH:.

# 4. Run Training
# We use python -m to run it as a module
python -m src.train

echo "Job finished at $(date)"