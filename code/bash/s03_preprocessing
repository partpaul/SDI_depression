#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --mem=40G
#SBATCH --array=1-3 # and when you are ready you write: 1-40
#SBATCH --output=/m/nbe/scratch/leap_mcpsych/fMRI/code/sample/ses-01/unwarped/fmriprep_%A_%a.out
#SBATCH --cpus-per-task=6


n=$SLURM_ARRAY_TASK_ID
iteration_raw=`sed -n "${n} p" controls.txt`	  # Get n-th line (1-indexed) of the file
iteration="${iteration_raw}"

mkdir -p /m/nbe/scratch/leap_mcpsych/fMRI/sample/ses-01/temp/${iteration}
echo ${iteration}

module load apptainer-fmriprep/25.1.4

apptainer_wrapper exec fmriprep /m/nbe/scratch/leap_mcpsych/fMRI/ses-01/raw_unwarped1/  /m/nbe/scratch/leap_mcpsych/fMRI/sample/ses-01/derivatives -w /m/nbe/scratch/leap_mcpsych/fMRI/sample/ses-01/temp/${iteration} participant --participant-label ${iteration} --n_cpus 6 --output-spaces MNI152NLin6Asym:res-2 T1w --cifti-output 91k --return-all-components --fd-spike-threshold 0.2 --fs-license-file /scratch/shareddata/set1/freesurfer/license.txt --write-graph






