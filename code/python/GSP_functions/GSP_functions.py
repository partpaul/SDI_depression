#!/usr/bin/env python
# coding: utf-8


# 19/05/25 Utils functions for GSP

import numpy as np
import numpy.linalg as la


def GSP_sub(W, graph_signal, frames):
    """ run GSP for each subject"""

    eigenvectors = find_harmonics(W)
    split_harmonic, low_freq_filter, low_freq_component, high_freq_filter, high_freq_component = signal_filtering(eigenvectors, graph_signal) 
    SDI,coupling_norm,decoupling_norm = compute_SDI(low_freq_component, high_freq_component, frames)
    BI  = compute_BI(low_freq_component, high_freq_component, graph_signal)
    
    return split_harmonic, low_freq_filter, high_freq_filter, SDI, coupling_norm, decoupling_norm, BI


def find_harmonics(W):
    """ Find SC harmonics using brain regions MEG order"""

    degree    = np.diag(np.power(np.sum(W, axis=1), -0.5)) # D^−1/2 
    laplacian = np.eye(W.shape[0]) - np.matmul(degree, np.matmul(W, degree))   # L = I - D^−1/2 AD^−1/2, # normalized laplacian from weight matrix W
    eigenvalues, eigenvectors = la.eigh(laplacian) # L = UΛU⊤ ==> GFT    # la.eigh instead of la.eig!! 16.07.25
    indexs    = np.argsort(eigenvalues)
    eigv_sort = eigenvalues[indexs]
    eigenvectors = eigenvectors[:, indexs]
    
    return eigenvectors
    

def signal_filtering(eigenvectors, graph_signal):
    """ Filter signal using low and high harmonics """

 # compute gpsd => project functional (signal_amplitude) data into the connectome harmonics (eigenvecs). In def fullpipeline, in utility_functions
    gft_coefficients = np.matmul(np.array(eigenvectors).T, graph_signal) #iGFT to obtain the Fourier coefficients and filter signal in the spectral domain
    psd_abs_squared  = np.abs(gft_coefficients)**2 # P(λ)=∣St(λ)∣2 the power or energy of the signal at each frequency 𝜆
    PSD  = np.mean(psd_abs_squared, axis=1) # average power spectral density across time points (?) ### TO CHECK ### 
    
    # To find the critical frequency split signal    
    halfpower = np.trapz(PSD) / 2 # instead of np.trapz(x=eigv_sort[:i],y=PSD[:I]) (methode 1)
    
    sum_of_freqs = 0
    i = 0
    while sum_of_freqs < halfpower:
        sum_of_freqs = np.trapz(PSD[:i])
        i += 1
    
    critical_freq = i - 1 # index of the last frequency that is below the half power.
    split_harmonic = critical_freq # harmonic splitting energy in half from low to high harmonics

    # split gft_coefficients => filter data in the spectral domain
    
    # Laplacian eigenvectors flipped in order (high frequencies first)
    eigenvectors_flip=np.fliplr(eigenvectors)
    cut_off = len(eigenvectors) - critical_freq
    
    high_freq_filter =  np.zeros(eigenvectors.shape)
    high_freq_filter[:,:cut_off] = eigenvectors_flip[ :, :cut_off]
    
    low_freq_filter = np.zeros(eigenvectors.shape)
    low_freq_filter[:,cut_off:]  = eigenvectors_flip[:, cut_off:]
    
    #filtering and inverse GFT
    gft_coefficients    = np.matmul(np.transpose(eigenvectors_flip),graph_signal)
    low_freq_component  = np.matmul(low_freq_filter, gft_coefficients)
    high_freq_component = np.matmul(high_freq_filter, gft_coefficients)
    

    return split_harmonic, low_freq_filter, low_freq_component, high_freq_filter, high_freq_component
    
    
def compute_SDI(low_freq_component, high_freq_component, frames):
    """ compute structural decoupling index (SDI) for each subject"""
    
    regions         = low_freq_component.shape[0] # number of regions
    coupling_norm   = np.zeros(regions) 
    decoupling_norm = np.zeros(regions)


    for j in range(regions):
         coupling_norm[j]  = np.linalg.norm(low_freq_component[j,frames]) # norm of the projected series in low graph frequency 
         decoupling_norm[j]= np.linalg.norm(high_freq_component[j,frames]) # norm of the projected series in high graph frequency       
      
    SDI = np.divide(decoupling_norm,coupling_norm) # emipirical individual SDI SDI = SDI[order_parcel]
    
    return SDI,coupling_norm,decoupling_norm

    
def compute_BI(low_freq_component, high_freq_component, graph_signal):
    """ compute Broadcasting index (BI) for each subject"""

    time_len = low_freq_component.shape[1] # number of time points
    # calculate power in time for normalization
    power_in_time = np.zeros([time_len])
    for t in range(time_len):
        power_in_time[t] = np.linalg.norm(graph_signal[:,t])
    power_in_time = np.transpose(power_in_time)
    
    X_c_norm = np.zeros(time_len)
    X_d_norm = np.zeros(time_len)
    
    # normalize backprojected time series
    for t in range(time_len):
    # normalize by the norm of the power of the original signal
        X_c_norm[t] = np.linalg.norm(low_freq_component[:,t])/power_in_time[t]  # ==> integration/coupling
        X_d_norm[t] = np.linalg.norm(high_freq_component[:,t])/power_in_time[t] # ==> segregation/decoupling 

    BI = X_d_norm-X_c_norm 
    
    return BI

def split_cp_dp_frames(subjects, BI_smth_all,Smooth_level):
    """ Compute segments of coupling and decoupling using the BDI """
    num_subs = len(subjects)
    total_cp = np.zeros(num_subs)  
    total_dp = np.zeros(num_subs) 
    frame_cp_idx_all   = {} 
    frame_dp_idx_all   = {} 
    size_frames_cp_all = {} 
    size_frames_dp_all = {}   

    for subidx, subject in enumerate(subjects): 
        BI = BI_smth_all[subidx][Smooth_level]     #BDI = BD_all[subidx]              
            
        # Coupling frames (fdata < 0)
        frame_cp                = BI < 0
        frame_cp_idx_all[subidx]= np.where(frame_cp)[0]
        total_cp[subidx]        = np.sum(frame_cp)

        
        # Decoupling frames (fdata > 0)
        frame_dp                = BI > 0
        frame_dp_idx_all[subidx]= np.where(frame_dp)[0]
        total_dp[subidx]        = np.sum(frame_dp)
    
            
        # compute size consecutuves cp and dp 
        size_frames_cp = []   
        size_frames_dp = [] 
    
        frames_BI       = frame_dp     # true = decoupling; false = coupling            
        current_BI_sign = frames_BI[0] #current_val = frames_BI[0] 
    
        count = 1            
        for i in range(1, len(frames_BI)):
            if frames_BI[i] == current_BI_sign:
                count += 1
            else:
                if current_BI_sign:
                    size_frames_cp.append(count)
                else:
                    size_frames_dp.append(count)
                current_BI_sign = frames_BI[i] # here it changes: if false ==> true, or viceversa
                count = 1
        
        # Add the final run
        if current_BI_sign:
            size_frames_cp.append(count)
        else:
            size_frames_dp.append(count)
            

    return {
        "total_cp" : total_cp,
        "frame_cp_idx_all": frame_cp_idx_all,
        "size_frames_cp_all": size_frames_cp_all,
        "total_dp" : total_dp,
        "frame_dp_idx_all": frame_dp_idx_all,
        "size_frames_dp_all": size_frames_dp_all      
    }


# def SDI_cp_dp_group(subjects, W, order_parcel, graph_signal_all, frames):
#     """ compute SDI across subjects for coupled and decoupled frames separately"""
#     n_subs          = len(subjects)
#     n_brain_regions = W.shape[0]
#     SDI_cp          = np.zeros((n_brain_regions,len(subjects)))
#     SDI_dp          = np.zeros((n_brain_regions,len(subjects)))


#     for subidx, subject in enumerate(subjects): 
#         graph_signal = graph_signal_all[subidx]
#         GSP_results  = GSP_sub(W, order_parcel, graph_signal)
        
#         split_harmonic_all[subidx] = GSP_results["split_harmonic"]
#         SDI_all[:,subidx]          = GSP_results["SDI"]
#         BI_all[subidx]             = GSP_results["BI"]

#     return {
#         "split_harmonic_all" : split_harmonic_all,
#         "SDI_all": SDI_all,
#         "BI_all": BI_all        
#     }

# def SDI_cp_dp_sub(W, order_parcel, graph_signal,frames):
#     """ compute SDI for specidic frames """

#     eigenvectors = find_harmonics(W,order_parcel) # function above
#     split_harmonic, low_freq_component, high_freq_component = signal_filtering(eigenvectors, graph_signal) # function above
#     SDI_= compute_SDI_CD(low_freq_component, high_freq_component, cp_frames)

#     return SDI
    
#  def compute_SDI_CD(low_freq_component, high_freq_component):
#     """ compute structural decoupling index (SDI) for each subject"""
    
#     regions         = low_freq_component.shape[0] # number of regions
#     coupling_norm   = np.zeros(regions) 
#     decoupling_norm = np.zeros(regions)


#     for j in range(regions):
#          coupling_norm[j]  = np.linalg.norm(low_freq_component[j,:]) # norm of the projected series in low graph frequency 
#          decoupling_norm[j]= np.linalg.norm(high_freq_component[j,:]) # norm of the projected series in high graph frequency       
      
#     SDI = np.divide(decoupling_norm,coupling_norm) # emipirical individual SDI SDI = SDI[order_parcel]
    
#     return SDI   
