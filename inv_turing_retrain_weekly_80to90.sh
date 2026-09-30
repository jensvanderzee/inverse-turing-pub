#!/bin/bash
#SBATCH --job-name=nca_training_adaptive_final_bice_7500_biglr_15to20
#SBATCH --output=nca_output_%j.txt
#SBATCH --error=nca_error_%j.txt
#SBATCH --time=48:00:00          # Set the maximum runtime
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --cpus-per-task=12        # Number of CPU cores per task
#SBATCH --mem=24G                 # Memory per node
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu


# Load necessary modules
module load 2024
module load Python/3.12.3

# Activate your Python environment if needed
# source /lustre/nobackup/WUR/ESG/zee034/PythonEnv/anunna_pinca/bin/activate

# Run your script
python /home/WUR/zee034/inverse-turing-testing/realdata_train_invPDE_80to90.py