#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 19/05/25 util functions for data


# In[ ]:


import numpy as np
import os
import os.path as op
import pandas as pd
import nibabel as nib
from nilearn import datasets
import numpy.linalg as la
import matplotlib.pyplot as plt
import itertools 
from numpy import linalg as LA
import nibabel as nib

# from brainspace.gradient import GradientMaps
# import warnings
# warnings.simplefilter('ignore')
# from brainspace.datasets import load_group_fc, load_parcellation, load_conte69
# from brainspace.plotting import plot_hemispheres
# from brainspace.utils.parcellation import map_to_labels
# import subprocess 


# In[ ]:


def compute_normalized_connectome(subs2use,filepath_DTI):

    """ compute structural adjancecy matrix (normalized using regions volume)
    original function in ComputeGroupConnectome.ipynb
    """
    # Load atlas
    atlas_schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=17, resolution_mm=1)
    
    # Load the atlas image (3D NIfTI)
    atlas_img = nib.load(atlas_schaefer.maps)
    atlas_data = atlas_img.get_fdata()
    
    # Get unique labels (excluding background 0)
    region_labels = np.unique(atlas_data)
    region_labels = region_labels[region_labels != 0]
    
    # Compute voxel size in mm³ (should be 1mm³ for resolution_mm=1)
    voxel_volume = np.prod(atlas_img.header.get_zooms())
    
    # Compute volumes
    region_volumes = np.zeros(len(region_labels))
    for i, label in enumerate(region_labels):
        voxel_count = np.sum(atlas_data == label)
        region_volumes[i] = voxel_count * voxel_volume  # in mm³
    
    # TO NORMALIZE: Build a matrix where each entry is the sum of volumes of the two regions
    volume_sum = region_volumes[:, None] + region_volumes[None, :]

    
    W_all = []
    for sub in subs2use:    
        filename_test = f"sub-CON{sub.split('-')[1]}" + '_parcels_coreg_yeo17_200.csv' 
        DTI_dir = op.join(filepath_DTI,filename_test)
        if op.exists(DTI_dir):
            Ws  = pd.read_csv(DTI_dir,header=None).values
            Ws = Ws[0:200,0:200]      # only cortical
            W_norm = Ws / volume_sum        
            W_all.append(W_norm)
        elif not op.exists(DTI_dir):  # check if the subject is in new added subjects 
            filepath_DTI_add = filepath_DTI + '/control_add'
            DTI_dir_add = op.join(filepath_DTI_add,filename_test)
            if op.exists(DTI_dir_add):
                Ws  = pd.read_csv(DTI_dir_add,header=None).values
                Ws = Ws[0:200,0:200]      # only cortical
                W_norm = Ws / volume_sum        
                W_all.append(W_norm)

        # compute group SC
        W_avGr = np.mean(np.array(W_all),0)  
        n_connectomes = len(W_all)


    return W_avGr, n_connectomes


def find_harmonics(W):
    """ Find SC harmonics using brain regions MEG order """

    degree    = np.diag(np.power(np.sum(W, axis=1), -0.5)) # D^−1/2 
    laplacian = np.eye(W.shape[0]) - np.matmul(degree, np.matmul(W, degree))   # L = I - D^−1/2 AD^−1/2, # normalized laplacian from weight matrix W
    eigenvalues, eigenvectors = la.eigh(laplacian) # L = UΛU⊤ ==> GFT    
    indexs    = np.argsort(eigenvalues)
    eigv_sort = eigenvalues[indexs]
    eigenvectors = eigenvectors[:, indexs]
    
    return eigenvectors


def smooth_BI(subjects,BI_all,size_win):
    BI_smth_all = {}
    for subidx in range(len(subjects)):
        BI = BI_all[subidx]
        BI_smth_sw = np.full((len(size_win), len(BI)), np.nan)  # To store smoothed signals for each window size
        
        # Apply smoothing for each window size
        for sw, window_size in enumerate(size_win):
            BI_smth_sw[sw] = np.convolve(BI, np.ones(window_size)/window_size, mode='same') # mode == same to keep data length
        
        # Store smoothed data for the current subject
        BI_smth_all[subidx] = BI_smth_sw
    
    return BI_smth_all

def SDI_Yeo(Reg_net_lab, SDI_all, subjects):
    num_sys    = len(np.unique(Reg_net_lab))
    len_subs   = len(subjects)
    
    within_sys_SDI = np.zeros((num_sys,len_subs))
    for sys_idx, sys in enumerate(np.unique(Reg_net_lab)):
            node_in_sys = np.where(Reg_net_lab == sys)[0]
            within_sys_SDI[sys_idx,:] = np.mean(np.log2(SDI_all[node_in_sys,:]),0)
    return within_sys_SDI

def FC_mat(bold_sig, idx_frame):
    """ compute zscore Pearson corr mat"""
    bold = bold_sig[:,idx_frame]
    PC   = np.corrcoef(bold, rowvar=True)
    np.fill_diagonal(PC, np.nan)  # Fisher Z is undefined for diagonal (r = 1)
    zFC  = np.arctanh(PC)
    np.fill_diagonal(zFC, 0)
    np.fill_diagonal(PC, 0)
    return PC, zFC


def Wei_zFC_mat_rescaled(X, w):
    """
    Compute a weighted Pearson correlation matrix from time series data,
    after rescaling the weight vector to the range [1, 2].

    Parameters:
    - X: np.ndarray of shape (T, N)
        Time series data with T time points and N regions
    - w: np.ndarray of shape (T,)
        Weights for each time point (non-negative)

    Returns:
    - corr: np.ndarray of shape (N, N)
        Weighted Pearson correlation matrix
    """
    # Ensure inputs are arrays
        # Ensure arrays
    X = np.asarray(X)
    w = np.asarray(w)

    # Rescale weights to [1, 2] without normalizing
    w_min, w_max = np.min(w), np.max(w)
    if w_max == w_min:
        w_rescaled = np.ones_like(w)
    else:
        w_rescaled = 1 + (w - w_min) / (w_max - w_min)

    # Compute the weighted mean 
    w_sum = np.sum(w_rescaled)
    mean_w = np.sum(X * w_rescaled[:, None], axis=0) / w_sum

    # Centered data
    X_centered = X - mean_w

    # Weighted covariance
    cov = (X_centered.T * w_rescaled) @ X_centered / w_sum

    # Weighted std dev
    var = np.sum((X_centered**2) * w_rescaled[:, None], axis=0) / w_sum
    std = np.sqrt(var)
    std[std == 0] = 1e-10

    # Correlation matrix
    corr = cov / np.outer(std, std)

    return corr

    
    
def fit_linear_map(A, B):
    # """To better grasp the diff between cp and dp
    # Finds W that minimizes ||Cp - W Dp||_F, returns residual and W
    # """
    B_pinv = np.linalg.pinv(B)
    W = A @ B_pinv
    residual = np.linalg.norm(A - W @ B, 'fro')
    return W, residual


def FC_within_Yeo(zFC, Reg_net_lab):
    num_sys        = len(np.unique(Reg_net_lab))
    within_sys_cor = np.zeros(len(np.unique(Reg_net_lab)))
    for sys in range(num_sys):
        node_in_sys    = np.where(Reg_net_lab == sys)[0] 
        hal_FC         = np.triu(zFC[np.ix_(node_in_sys,node_in_sys)],k=1)
        upper_vals     = hal_FC[np.triu_indices_from(hal_FC, k=1)]
        within_sys_cor[sys] = np.mean(upper_vals)
    return within_sys_cor

def FC_between_Yeo(zFC, Reg_net_lab):
    num_sys    = len(np.unique(Reg_net_lab))
    nets       = list(range(0, num_sys))
    net_pairs  = list(itertools.combinations(nets, 2))
    between_sys_cor = np.zeros(len(net_pairs))
    bt_idx=0
    for btw_net in net_pairs:
        i, j = btw_net
        node_in_sys1    = np.where(Reg_net_lab == i)[0]
        node_in_sys2    = np.where(Reg_net_lab == j)[0]
        hal_FC = np.triu(zFC[np.ix_(node_in_sys1,node_in_sys2)],k=1)
        upper_vals      = hal_FC[np.triu_indices_from(hal_FC, k=1)]
        between_sys_cor[bt_idx] = np.mean(upper_vals)
        bt_idx+=1
    return between_sys_cor

# SIMILARITY METRICS

class distance_FC(object):
    def __init__(self, FC1, FC2, eig_thresh=10**(-3)):
        self.FC1 = FC1
        self.FC2 = FC2
        self.eig_thresh = eig_thresh

        # ensure symmetric
        self.FC1 = self._ensure_symmetric(self.FC1)
        self.FC2 = self._ensure_symmetric(self.FC2)

    def _info(self, s):
        print('INFO: %s' % s)

    def _ensure_symmetric(self, Q):
        '''
        computation is sometimes not precise (round errors),
        so ensure matrices that are supposed to be
        symmetric are symmetric
        '''
        return (Q + np.transpose(Q))/2

    def _vectorize(self, Q):
        '''
        given a symmetric matrix (FC), return unique
        elements as an array. Ignore diagonal elements
        '''
        # extract lower triangular matrix
        tri = np.tril(Q, -1)

        vec = []
        for ii in range(1, tri.shape[0]):
            for jj in range(ii):
                vec.append(tri[ii, jj])
        
        return np.asarray(vec)

    def geodesic(self):
        '''
        dist = sqrt(trace(log^2(M)))
        M = Q_1^{-1/2}*Q_2*Q_1^{-1/2}
        '''
        # compute Q_1^{-1/2} via eigen value decmposition
        u, s, _ = LA.svd(self.FC1, full_matrices=True)

        ## lift very small eigen values
        for ii, s_ii in enumerate(s):
            if s_ii < self.eig_thresh:
                s[ii] = self.eig_thresh

        '''
        since FC1 is in S+, u = v, u^{-1} = u'
        FC1 = usu^(-1)
        FC1^{1/2} = u[s^{1/2}]u'
        FC1^{-1/2} = u[s^{-1/2}]u'
        '''
        FC1_mod = u @ np.diag(s**(-1/2)) @ np.transpose(u)
        M = FC1_mod @ self.FC2 @ FC1_mod

        '''
        trace = sum of eigenvalues;
        np.logm might have round errors,
        implement using svd instead
        '''
        _, s, _ = LA.svd(M, full_matrices=True)

        return np.sqrt(np.sum(np.log(s)**2))

    def pearson(self):
        '''
        conventional Pearson distance between
        two FC matrices. The matrices are vectorized
        '''
        vec1 = self._vectorize(self.FC1)
        vec2 = self._vectorize(self.FC2)

        return (1 - np.corrcoef(vec1, vec2)[0, 1])/2

def SFS(BOLD_sig, idx_frame):
    br_state = np.mean(BOLD_sig[:,idx_frame],1)
    return br_state

def wSFS(BOLD_sig, w):
    # Rescale weights to [1, 2] without normalizing
    w_min, w_max = np.min(w), np.max(w)
    if w_max == w_min:
        w_rescaled = np.ones_like(w)
    else:
        w_rescaled = 1 + (w - w_min) / (w_max - w_min)

    # Compute the weighted mean 
    w_sum = np.sum(w_rescaled)    
    weighted_avg = np.sum(BOLD_sig * w_rescaled, axis=1)/w_sum  # shape: (n,)
    return weighted_avg

def Yeo_states(BOLD_sig,Net_label):
    time_points     = BOLD_sig.shape[1]
    num_sys         = len(np.unique(Net_label))
    n_brain_regions = BOLD_sig.shape[0]
    
    sys_av_BOLD = np.zeros((num_sys,time_points))
    for sys_idx, sys in enumerate(np.unique(Net_label)):
            node_in_sys = np.where(Net_label == sys)[0]
            sys_av_BOLD[sys_idx,:] = np.mean(BOLD_sig[node_in_sys,:],0)
        
    max_amp = np.argmax(sys_av_BOLD,0) # find most activated yeo net for each time point
    
    br_state_sys   = np.zeros((n_brain_regions,num_sys))
    for sys_idx, sys in enumerate(np.unique(Net_label)):
        br_state_sys[:,sys_idx] = np.mean(BOLD_sig[:,max_amp == sys_idx],1)

    return br_state_sys



def FC_group_gradients(nb_comp,kernel,FC_mat_gr):
    """
    using https://github.com/Bronte-      Mckeown/GradientAnalysis/blob/master/calculate_gradients.py
    """

    #set number of components
    NOcomp = nb_comp
    
    #set kernel
    kernel = kernel

    #load HCP data from BrainSpace and calculate group level gradients
    conn_matrix   = load_group_fc('schaefer', scale=200)
    conn_gradient = GradientMaps(n_components=NOcomp, kernel= kernel)
    conn_gradient.fit(conn_matrix)

    #load FC matrix and calculate gradients
    m = FC_mat_gr
    group_gradient = GradientMaps(n_components=NOcomp, kernel= kernel, alignment='procrustes')
    #align our group level gradients to the HCP gradients
    group_aligned = group_gradient.fit([m], reference = conn_gradient.gradients_)

    return group_aligned, conn_gradient



def FC_sub_gradients(nb_comp,kernel,group_gradient, FC_sub):
    """
    using https://github.com/Bronte-      Mckeown/GradientAnalysis/blob/master/calculate_gradients.py
    """

    #set number of components
    NOcomp = nb_comp
    
    #set kernel
    kernel = kernel

    #Load the correlation matrix for that individual and calculate individual gradients, aligned to group
    ind_matrix = FC_sub
    ind_gradient = GradientMaps(n_components=10, kernel= kernel, alignment='procrustes')
    ind_gradient.fit([ind_matrix],reference = group_gradient.aligned_[0])
    sub_gradient = ind_gradient.aligned_[0]

    return sub_gradient












