#!/bin/bash
#SBATCH --job-name=nnunet_batch
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=def-punithak
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --array=1-100       # Adjust to number of batches

module load cuda           # or your relevant modules

# Activate local virtual environment
source /home/ranashah/scratch/MBH-SEG-2024-winning-solution/.venv/bin/activate

# Set paths for batch input and output folders
BATCH_PARENT="/home/ranashah/scratch/batches"
OUTPUT_PARENT="/home/ranashah/scratch/output"

# Find batch folder for this SLURM task ID
BATCH_FOLDER=$(ls $BATCH_PARENT | sed -n "${SLURM_ARRAY_TASK_ID}p")

echo "Processing batch folder: $BATCH_FOLDER"

# Run submission.py from project directory
python /home/ranashah/scratch/MBH-SEG-2024-winning-solution/submission.py --input_folder $BATCH_PARENT/$BATCH_FOLDER --output_folder $OUTPUT_PARENT/$BATCH_FOLDER
