#!/bin/bash -l
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --output=/m/nbe/scratch/leap_mcpsych/fMRI/code/sample/ses-01/unwarped/unwarped_%A_%a.out
#SBATCH --cpus-per-task=1
#SBATCH --array=1-5

#for n in $(find /scratch/nbe/leap_mcpsych/fMRI/ses-01/raw -maxdepth 1 -type d  -name "*sub*"); do
#	echo unwarp_bids_folder $n  /scratch/nbe/leap_mcpsych/fMRI/ses-01/raw_unwarped
#done

module load mamba
source activate /scratch/nbe/leap_mcpsych/fMRI/code/git/partanp5/leap_fMRI/code/env
source /scratch/nbe/leap_mcpsych/fMRI/code/git/partanp5/leap_fMRI/code/bash/lib.sh
module load fsl
source $FSLDIR/etc/fslconf/fsl.sh

# InputFolder=$(find /scratch/nbe/leap_mcpsych/fMRI/ses-01/raw -maxdepth 1 -type d  -name "*sub*"|sort|sed -n "${SLURM_ARRAY_TASK_ID} p")

n=$SLURM_ARRAY_TASK_ID
InputFolder=`sed -n "${n} p" extra.txt`	  # Get n-th line (1-indexed) of the file


unwarp_bids_folder /scratch/nbe/leap_mcpsych/fMRI/ses-01/raw/$InputFolder  /scratch/nbe/leap_mcpsych/fMRI/ses-01/raw_unwarped
