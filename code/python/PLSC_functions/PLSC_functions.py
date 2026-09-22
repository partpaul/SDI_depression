# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 2024

@author: Wenya Liu
"""

import numpy as np
import pandas as pd

import random
from scipy.stats import zscore
from tqdm import tqdm

from sklearn.linear_model import LinearRegression

def PLSC(imag_matrix, beh_matrix, z_score=True, ICA_convention=True):

      if len(imag_matrix.shape) >= 2:
            imag_matrix = imag_matrix.reshape(imag_matrix.shape[0], -1)
      if z_score:
            Y = zscore(beh_matrix, 0, ddof=1, nan_policy='omit') # 1: z-score across features, 0: z-score across subjects
            X = zscore(imag_matrix, 0, ddof=1, nan_policy='omit')
      else:
            Y = beh_matrix
            X = imag_matrix
      # svd
      R = np.dot(Y.T, X)
      U, S, Vt = np.linalg.svd(R, full_matrices=False)

      #### adding ICA convention
      n_comp = Y.shape[1]
      if ICA_convention:
            for i in range(n_comp):
                  max_id = np.argmax(np.abs(Vt[i,:]))
                  if Vt[i,max_id] < 0:
                        Vt[i,:] = -Vt[i,:]
                        U[:,i] = -U[:,i]

      # score
      Lx = np.dot(X, Vt.T)
      Ly = np.dot(Y, U)
      # loading
      n_fea_X = X.shape[1]
      n_fea_Y = Y.shape[1]
      corr_Lx_X = np.corrcoef(X, Lx, rowvar=False)[:n_fea_X,-n_comp:]
      corr_Ly_Y = np.corrcoef(Y, Ly, rowvar=False)[:n_fea_Y,-n_comp:]

      return Lx, Ly, corr_Lx_X, corr_Ly_Y, S, U, Vt

def rri_bootprocrust(origlv,bootlv):
      # define coordinate space between original and bootstrap LVs
      temp = np.dot(origlv.T, bootlv)
      U,S,Vt = np.linalg.svd(temp, full_matrices=False)
      # determine procrustean transform
      rotatemat= np.dot(U, Vt)

      return rotatemat

def permutation_test(imag_matrix, beh_matrix, S, U, n_perm = 1000):

      if len(imag_matrix.shape) >= 2:
            imag_matrix = imag_matrix.reshape(imag_matrix.shape[0], -1)
            
      Y = zscore(beh_matrix, 0, ddof=1, nan_policy='omit')
      X = zscore(imag_matrix, 0, ddof=1, nan_policy='omit')
      Sp_vect = np.zeros([n_perm, Y.shape[1]])

      for n_p in tqdm(range(n_perm)):
            n_sub = X.shape[0]
            ind = np.array(range(n_sub))
            random.shuffle(ind)
            Yp = Y
            Xp = X[ind,:]

            Rp = Yp.T @ Xp
            Up, Sp, Vpt = np.linalg.svd(Rp, full_matrices=False)
            Sp = np.diag(Sp)  # transform it to a diagonal matrix
            rotatemat = rri_bootprocrust(U,Up)
            Up = Up @ Sp @ rotatemat
            Vp = Vpt.T @ Sp @ rotatemat
            Sp = np.sqrt(np.sum(Vp**2, axis=0))   # the same as using np.sqrt(np.sum(Up**2, axis=0))
            Sp_vect[n_p,:] = Sp

            # Compute the p-values from the permutation null distribution
            sp = np.sum(Sp_vect >= S, 0)
            LC_pvals = sp/n_perm

      return LC_pvals

def confounds_regression(X, Y):
      # remove effect of Y from each feature in X
      shape1   = X.shape
      n_subj   = shape1[0]
      n_feat   = np.prod(shape1[1:])
      X_mat = X.reshape(n_subj,n_feat)    
      X_resdual = X_mat.copy()
      model = LinearRegression()
      for n in range(n_feat):
            x = X_mat[:,n]
            model.fit(Y, x)
            x_pre = model.predict(Y)
            X_resdual[:,n] = x-x_pre
      X_resdual = np.reshape(X_resdual,shape1)
      return X_resdual


def bootstrap_stats(results_p):

      results_bootstrap_stats = dict()
      n_bs = results_p['Up_vect'].shape[0]
      # Computing mean, std and CIs for bootstrap saliences
      results_bootstrap_stats['Up_mean'] = np.mean(results_p['Up_vect'],0)
      results_bootstrap_stats['Up_sem'] = np.std(results_p['Up_vect'], axis=0, ddof=1) / np.sqrt(n_bs)
      results_bootstrap_stats['Up_lB'] = np.percentile(results_p['Up_vect'],2.5,0)
      results_bootstrap_stats['Up_uB'] = np.percentile(results_p['Up_vect'],97.5,0)

      results_bootstrap_stats['Vp_mean'] = np.mean(results_p['Vp_vect'],0)
      results_bootstrap_stats['Vp_sem'] = np.std(results_p['Vp_vect'], axis=0, ddof=1) / np.sqrt(n_bs)
      results_bootstrap_stats['Vp_lB'] = np.percentile(results_p['Vp_vect'],2.5,0)
      results_bootstrap_stats['Vp_uB'] = np.percentile(results_p['Vp_vect'],97.5,0)

      # Computing mean, std and CIs for bootstrap loadings
      results_bootstrap_stats['corr_Lx_Xp_mean'] = np.mean(results_p['corr_Lx_Xp'],0)
      results_bootstrap_stats['corr_Lx_Xp_sem'] = np.std(results_p['corr_Lx_Xp'], axis=0, ddof=1) / np.sqrt(n_bs)
      results_bootstrap_stats['corr_Lx_Xp_lB'] = np.percentile(results_p['corr_Lx_Xp'],2.5,0)
      results_bootstrap_stats['corr_Lx_Xp_uB'] = np.percentile(results_p['corr_Lx_Xp'],97.5,0)

      results_bootstrap_stats['corr_Ly_Yp_mean'] = np.mean(results_p['corr_Ly_Yp'],0)
      results_bootstrap_stats['corr_Ly_Yp_sem'] = np.std(results_p['corr_Ly_Yp'], axis=0, ddof=1) / np.sqrt(n_bs)
      results_bootstrap_stats['corr_Ly_Yp_lB'] = np.percentile(results_p['corr_Ly_Yp'],2.5,0)
      results_bootstrap_stats['corr_Ly_Yp_uB'] = np.percentile(results_p['corr_Ly_Yp'],97.5,0)

      return results_bootstrap_stats



def bootstrap_test(imag_matrix, beh_matrix, U, V, n_perm = 1000, procrustas='U_only'):

      # getting bootstrap orders
      n_subjects = imag_matrix.shape[0]
      bootstrap_samples = []
      for _ in range(n_perm):
            resample_indices = np.random.choice(range(n_subjects), size=n_subjects, replace=True)
            bootstrap_samples.append(resample_indices)
      bootstrap_samples = np.array(bootstrap_samples)
      # business as usual
      if len(imag_matrix.shape) == 3:
            imag_matrix = imag_matrix.reshape(imag_matrix.shape[0], -1)
            
      results_p = {
            # 'Lxp': np.ndarray((n_perm, n_subjects, beh_matrix.shape[1]), dtype='float64'),
            # 'Lyp': np.ndarray((n_perm, n_subjects, beh_matrix.shape[1]), dtype='float64'),
            'Up_vect': np.ndarray((n_perm, *U.shape), dtype='float64'),
            'Vp_vect': np.ndarray((n_perm, *V.shape), dtype='float64'),
            'corr_Lx_Xp': np.ndarray((n_perm, imag_matrix.shape[1], beh_matrix.shape[1]), dtype='float64'),
            'corr_Ly_Yp': np.ndarray((n_perm, beh_matrix.shape[1], beh_matrix.shape[1]), dtype='float64')
            }
      for n_p in tqdm(range(n_perm)):
            Y = zscore(beh_matrix[bootstrap_samples[n_p]], 0, ddof=1, nan_policy='omit')
            X = zscore(imag_matrix[bootstrap_samples[n_p]], 0, ddof=1, nan_policy='omit')
            
            Rp = Y.T @ X
            Up, Sp, Vpt = np.linalg.svd(Rp, full_matrices=False)
            
            if procrustas == 'U_only':
                  rotatemat_full = rri_bootprocrust(U, Up)
                  Vp = Vpt.T @ np.diag(Sp) @ rotatemat_full
                  Up = Up @ np.diag(Sp) @ rotatemat_full

                  Vp = Vp / Sp
                  Up = Up / Sp
            else:
                  rotatemat1 = rri_bootprocrust(U, Up)
                  rotatemat2 = rri_bootprocrust(V, Vpt.T)
                  rotatemat_full = (rotatemat1 + rotatemat2)/2

                  Vp = Vpt.T @ rotatemat_full
                  Up = Up @ rotatemat_full

            results_p['Up_vect'][n_p] = Up
            results_p['Vp_vect'][n_p] = Vp

            n_fea_X = X.shape[1]
            n_fea_Y = Y.shape[1]
            n_comp = Y.shape[1]
            Lxp = np.dot(X, Vp)
            Lyp = np.dot(Y, Up)
            corr_Lx_Xp = np.corrcoef(X, Lxp, rowvar=False)[:n_fea_X,-n_comp:]
            corr_Ly_Yp = np.corrcoef(Y, Lyp, rowvar=False)[:n_fea_Y,-n_comp:]

            # results_p['Lxp'][n_p] = Lxp
            # results_p['Lyp'][n_p] = Lyp
            results_p['corr_Lx_Xp'][n_p] = corr_Lx_Xp
            results_p['corr_Ly_Yp'][n_p] = corr_Ly_Yp

      return results_p
