#!/bin/bash
#SBATCH --job-name=7th_run              # Job name
#SBATCH --account=genacc_q              # Use the general account
#SBATCH --mail-type=BEGIN,END           # Mail notifications for start and end of job
#SBATCH --mail-user=mi21b@fsu.edu       # Your email for notifications
#SBATCH --nodes=1                       # Number of nodes (1 for single-node computation)
#SBATCH --ntasks=16                     # Total number of tasks (cores to use)
#SBATCH --time=2-00:00:00               # Max runtime (1 day)
#SBATCH --mem=32G                       # Memory per node (adjust based on requirements)
#SBATCH --output=main_output.log         # Output log file
#SBATCH --error=main_error.log           # Error log file

# Load required modules
module load anaconda/3.11.5             # Load Anaconda

# Activate your Python environment
source /gpfs/home/mi21b/Research/extra_exp/extra-env/bin/activate

# Change to the directory where your script is located
cd "/gpfs/home/mi21b/Research/extra_exp/7th_run_nov27_2024"

# Execute your Python script
python main.py
