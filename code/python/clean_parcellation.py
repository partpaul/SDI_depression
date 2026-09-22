import sys
import nilearn
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image as img
from nilearn import regions
from nilearn import datasets, maskers
from nilearn.maskers import NiftiMasker
from nilearn.maskers import NiftiLabelsMasker
import bottleneck
import os


# Parameters, choose weather with or without global signal

confound_list =  ['trans_x', 'trans_x_derivative1', 'trans_x_power2',
      'trans_x_derivative1_power2', 'trans_y', 'trans_y_derivative1',
      'trans_y_power2', 'trans_y_derivative1_power2', 'trans_z',
      'trans_z_derivative1', 'trans_z_derivative1_power2', 'trans_z_power2',
      'rot_x', 'rot_x_derivative1', 'rot_x_power2',
      'rot_x_derivative1_power2', 'rot_y', 'rot_y_derivative1',
      'rot_y_power2', 'rot_y_derivative1_power2', 'rot_z',
      'rot_z_derivative1', 'rot_z_derivative1_power2', 'rot_z_power2',
      'csf', 'csf_derivative1',
      'csf_power2', 'csf_derivative1_power2', 'white_matter',
      'white_matter_derivative1', 'white_matter_power2',
      'white_matter_derivative1_power2', 
      'global_signal', 'global_signal_derivative1', 'global_signal_power2', 'global_signal_derivative1_power2']

high_pass = 0.01
low_pass = 0.1

# Session and subject

ses = 'ses-01'
subject = sys.argv[1]

# Create parcellated data

#func_filename = '/m/nbe/scratch/leap_mcpsych/fMRI/sample/ses-01/ICA_FIX/' + subject + '/filtered_func_data/filtered_func_data.ica/filtered_func_data_clean.nii.gz'
filepath = '/m/nbe/scratch/leap_mcpsych/fMRI/sample/' + ses + '/derivatives/' + subject + '/func/'
func_filename = subject + '_task-rest_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz'
conf_filename = subject + '_task-rest_desc-confounds_timeseries.tsv'

func_img = img.load_img(filepath+func_filename)
confounds = pd.read_csv(filepath+conf_filename, delimiter='\t')

confounds_reg = confounds[confound_list]
confounds_reg = confounds_reg.fillna(0, inplace=False)
confounds_matrix = confounds_reg.values

clean_imgs = img.clean_img(func_img, confounds=confounds_matrix, low_pass=low_pass, high_pass=high_pass, t_r=1.25, standardize='zscore_sample')

atlas = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=17, resolution_mm=1)
atlas_img = atlas.maps  

masker = nilearn.maskers.NiftiLabelsMasker(labels_img=atlas_img, standardize=False)
parcellated_data = masker.fit_transform(clean_imgs)

save_path = '/m/nbe/scratch/leap_mcpsych/fMRI/' + ses + '/parcel_data/' + subject + '/Schaefer/'
save_name = subject + '_task-rest_bold.npy'
np.save(save_path+save_name, parcellated_data)
