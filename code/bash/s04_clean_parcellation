#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-5
#SBATCH --output=/m/nbe/scratch/leap_mcpsych/fMRI/code/out/ses-01/parcellation_%A_%a.out

# Activate your environment
module load mamba
source activate /m/nbe/scratch/leap_mcpsych/Paula/env/leap_fmri

# Read subject from list
n=$SLURM_ARRAY_TASK_ID
subject=$(sed -n "${n}p" extra.txt)

# Run your Python script
/m/nbe/scratch/leap_mcpsych/Paula/env/leap_fmri/bin/python ./python/clean_parcellation.py $subject
