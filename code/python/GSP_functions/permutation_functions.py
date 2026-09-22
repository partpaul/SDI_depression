#!/usr/bin/env python
# coding: utf-8


# 19/05/25 Utils functions for permutations

import numpy as np
import random
from itertools import groupby
import numpy.linalg as la

def get_runs(labels):
    """Get runs of labels with their lengths."""
    return [(label, sum(1 for _ in group)) for label, group in groupby(labels)]

def extract_run_lengths(labels):
    """Separate run lengths for 1s and 0s."""
    runs = get_runs(labels)
    ones_runs = [length for label, length in runs if label == 1]
    zeros_runs = [length for label, length in runs if label == 0]
    return ones_runs, zeros_runs

def build_sequence(ones_runs, zeros_runs):
    """Build a new label sequence from shuffled run lengths."""
    random.shuffle(ones_runs)
    random.shuffle(zeros_runs)

    sequence = []
    current_label = random.choice([1, 0])  # Random starting label

    while ones_runs or zeros_runs:
        if current_label == 1 and ones_runs:
            length = ones_runs.pop()
            sequence.extend([1] * length)
            current_label = 0
        elif current_label == 0 and zeros_runs:
            length = zeros_runs.pop()
            sequence.extend([0] * length)
            current_label = 1
        else:
            # Append remaining runs of the other label
            remaining_runs = ones_runs if ones_runs else zeros_runs
            label = 1 if ones_runs else 0
            for length in remaining_runs:
                sequence.extend([label] * length)
            break
    return sequence


def permutation_length(original_labels):
    """
    Generate permutation of a binary label sequence and return as NumPy array.
    
    Returns:
        list with permuted values
    """
    ones_runs, zeros_runs = extract_run_lengths(original_labels)
    total_length = len(original_labels)    
    new_seq = build_sequence(ones_runs[:], zeros_runs[:])  # Use copies
    if len(new_seq) != total_length:
        raise ValueError("Generated sequence has incorrect length.")
    permutation = new_seq

    return permutation
    
def permutations_length(original_labels, n_permutations):
    """
    Generate permutations of a binary label sequence and return as NumPy array.
    
    Returns:
        np.ndarray of shape (n_permutations, len(original_labels))
    """
    ones_runs, zeros_runs = extract_run_lengths(original_labels)
    total_length = len(original_labels)
    permutations = np.empty((n_permutations, total_length), dtype=int)

    for i in range(n_permutations):
        new_seq = build_sequence(ones_runs[:], zeros_runs[:])  # Use copies
        if len(new_seq) != total_length:
            raise ValueError("Generated sequence has incorrect length.")
        permutations[i, :] = new_seq

    return permutations

# permutation structural connectome

# from https://github.com/aestrivex/bctpy/blob/master/bct/utils/miscellaneous_utilities.py

def get_rng(seed=None):
    """
    By default, or if `seed` is np.random, return the global RandomState
    instance used by np.random.
    If `seed` is a RandomState instance, return it unchanged.
    Otherwise, use the passed (hashable) argument to seed a new instance
    of RandomState and return it.

    Parameters
    ----------
    seed : hashable or np.random.RandomState or np.random, optional

    Returns
    -------
    np.random.RandomState
    """
    if seed is None or seed == np.random:
        return np.random.mtrand._rand
    elif isinstance(seed, np.random.RandomState):
        return seed
    try:
        rstate =  np.random.RandomState(seed)
    except ValueError:
        rstate = np.random.RandomState(random.Random(seed).randint(0, 2**32-1))
    return rstate

# from https://github.com/aestrivex/bctpy/blob/master/bct/algorithms/reference.py

def randmio_und_signed(R, itr, seed=None):
    '''
    This function randomizes an undirected weighted network with positive
    and negative weights, while simultaneously preserving the degree
    distribution of positive and negative weights. The function does not
    preserve the strength distribution in weighted networks.

    Parameters
    ----------
    W : NxN np.ndarray
        undirected binary/weighted connection matrix
    itr : int
        rewiring parameter. Each edge is rewired approximately itr times.
    seed : hashable, optional
        If None (default), use the np.random's global random state to generate random numbers.
        Otherwise, use a new np.random.RandomState instance seeded with the given value.

    Returns
    -------
    R : NxN np.ndarray
        randomized network
    eff : int
        number of rewirings made
    '''
    rng = get_rng(seed)
    R = R.copy()
    n = len(R)

    itr *= int(n * (n -1) / 2)

    max_attempts = int(np.round(n / 2))
    eff = 0

    for it in range(int(itr)):
        att = 0
        while att <= max_attempts:

            a, b, c, d = pick_four_unique_nodes_quickly(n, rng)

            r0_ab = R[a, b]
            r0_cd = R[c, d]
            r0_ad = R[a, d]
            r0_cb = R[c, b]

            #rewiring condition
            if (    np.sign(r0_ab) == np.sign(r0_cd) and
                    np.sign(r0_ad) == np.sign(r0_cb) and
                    np.sign(r0_ab) != np.sign(r0_ad)):

                R[a, d] = R[d, a] = r0_ab
                R[a, b] = R[b, a] = r0_ad

                R[c, b] = R[b, c] = r0_cd
                R[c, d] = R[d, c] = r0_cb

                eff += 1
                break

            att += 1

    return R, eff

# from https://github.com/aestrivex/bctpy/blob/master/bct/algorithms/reference.py

def null_model_und_sign(W, bin_swaps=5, wei_freq=.1, seed=None):
    '''
    This function randomizes an undirected network with positive and
    negative weights, while preserving the degree and strength
    distributions. This function calls randmio_und.m

    Parameters
    ----------
    W : NxN np.ndarray
        undirected weighted connection matrix
    bin_swaps : int
        average number of swaps in each edge binary randomization. Default
        value is 5. 0 swaps implies no binary randomization.
    wei_freq : float
        frequency of weight sorting in weighted randomization. 0<=wei_freq<1.
        wei_freq == 1 implies that weights are sorted at each step.
        wei_freq == 0.1 implies that weights sorted each 10th step (faster,
            default value)
        wei_freq == 0 implies no sorting of weights (not recommended)
    seed : hashable, optional
        If None (default), use the np.random's global random state to generate random numbers.
        Otherwise, use a new np.random.RandomState instance seeded with the given value.

    Returns
    -------
    W0 : NxN np.ndarray
        randomized weighted connection matrix
    R : 4-tuple of floats
        Correlation coefficients between strength sequences of input and
        output connection matrices, rpos_in, rpos_out, rneg_in, rneg_out

    Notes
    -----
    The value of bin_swaps is ignored when binary topology is fully
        connected (e.g. when the network has no negative weights).
    Randomization may be better (and execution time will be slower) for
        higher values of bin_swaps and wei_freq. Higher values of bin_swaps
        may enable a more random binary organization, and higher values of
        wei_freq may enable a more accurate conservation of strength
        sequences.
    R are the correlation coefficients between positive and negative
        strength sequences of input and output connection matrices and are
        used to evaluate the accuracy with which strengths were preserved.
        Note that correlation coefficients may be a rough measure of
        strength-sequence accuracy and one could implement more formal tests
        (such as the Kolmogorov-Smirnov test) if desired.
    '''
    rng = get_rng(seed)
    if not np.allclose(W, W.T):
        raise BCTParamError("Input must be undirected")
    W = W.copy()
    n = len(W)
    np.fill_diagonal(W, 0)  # clear diagonal
    Ap = (W > 0)  # positive adjmat
    An = (W < 0)  # negative adjmat

    if np.size(np.where(Ap.flat)) < (n * (n - 1)):
        W_r, eff = randmio_und_signed(W, bin_swaps, seed=rng)
        Ap_r = W_r > 0
        An_r = W_r < 0
    else:
        Ap_r = Ap
        An_r = An

    W0 = np.zeros((n, n))
    for s in (1, -1):
        if s == 1:
            Acur = Ap
            A_rcur = Ap_r
        else:
            Acur = An
            A_rcur = An_r

        S = np.sum(W * Acur, axis=0)  # strengths
        Wv = np.sort(W[np.where(np.triu(Acur))])  # sorted weights vector
        i, j = np.where(np.triu(A_rcur))
        Lij, = np.where(np.triu(A_rcur).flat)  # weights indices

        P = np.outer(S, S)

        if wei_freq == 0:  # get indices of Lij that sort P
            Oind = np.argsort(P.flat[Lij])  # assign corresponding sorted
            W0.flat[Lij[Oind]] = s * Wv  # weight at this index
        else:
            wsize = np.size(Wv)
            wei_period = np.round(1 / wei_freq).astype(int)  # convert frequency to period
            lq = np.arange(wsize, 0, -wei_period, dtype=int)
            for m in lq:  # iteratively explore at this period
                # get indices of Lij that sort P
                Oind = np.argsort(P.flat[Lij])
                R = rng.permutation(m)[:np.min((m, wei_period))]
                for q, r in enumerate(R):
                    # choose random index of sorted expected weight
                    o = Oind[r]
                    W0.flat[Lij[o]] = s * Wv[r]  # assign corresponding weight

                    # readjust expected weighted probability for i[o],j[o]
                    f = 1 - Wv[r] / S[i[o]]
                    P[i[o], :] *= f
                    P[:, i[o]] *= f
                    f = 1 - Wv[r] / S[j[o]]
                    P[j[o], :] *= f
                    P[:, j[o]] *= f

                    # readjust strength of i[o]
                    S[i[o]] -= Wv[r]
                    # readjust strength of j[o]
                    S[j[o]] -= Wv[r]

                O = Oind[R]
                # remove current indices from further consideration
                Lij = np.delete(Lij, O)
                i = np.delete(i, O)
                j = np.delete(j, O)
                Wv = np.delete(Wv, R)

    W0 = W0 + W0.T

    rpos_in = np.corrcoef(np.sum(W * (W > 0), axis=0),
                          np.sum(W0 * (W0 > 0), axis=0))
    rpos_ou = np.corrcoef(np.sum(W * (W > 0), axis=1),
                          np.sum(W0 * (W0 > 0), axis=1))
    rneg_in = np.corrcoef(np.sum(-W * (W < 0), axis=0),
                          np.sum(-W0 * (W0 < 0), axis=0))
    rneg_ou = np.corrcoef(np.sum(-W * (W < 0), axis=1),
                          np.sum(-W0 * (W0 < 0), axis=1))
    return W0, (rpos_in[0, 1], rpos_ou[0, 1], rneg_in[0, 1], rneg_ou[0, 1])
    

def configuration_model(W):
    """
    Generate random SC using the configuration model (CM). Return ordered and flipped (from high to low) harmonics of the CM 
    """
    n_brain_regions     = W.shape[0]
    # Normalized SC 
    degree = np.diag(np.power(np.sum(W, axis=1), -0.5))  # D^(-1/2)
    W_new = degree @ W @ degree  # D^(-1/2) * W_avGr * D^(-1/2)
    # Compute expected weights under the configuration model
    s = np.sum(W_new, axis=1)  # Strength sequence (sum of weights per node)
    W_r = np.outer(s, s) / np.sum(s)  # Configuration model matrix

    #laplacian
    L_r    = np.diag(np.sum(W_new,1))-W_r;
    eigenvalues, eigenvectors = la.eigh(L_r) # L = UΛU⊤ ==> GFT    
    indexs    = np.argsort(eigenvalues)
    #eigv_sort = eigenvalues[indexs]
    eigenvectors = eigenvectors[:, indexs]
    eigenvectors_rand=np.fliplr(eigenvectors)  


    return eigenvectors_rand



def conf_model_degree_rand(W):
    """
    Generate a configuration model with randomized node degrees 
    (i.e., preserve global degree distribution, but shuffle node-specific degrees).
    
    Returns harmonics (eigenvectors) of the degree-randomized Laplacian.
    """
    n = W.shape[0]

    # Step 1: Normalize W (optional, as in original)
    degree_mat = np.diag(np.power(np.sum(W, axis=1), -0.5))  # D^(-1/2)
    W_norm = degree_mat @ W @ degree_mat

    # Step 2: Compute node strengths (degree-like weights)
    s = np.sum(W_norm, axis=1)  # strength per node

    # Step 3: Shuffle the strength sequence
    s_shuffled = np.random.permutation(s)

    # Step 4: Generate expected weight matrix (configuration model)
    W_rand = np.outer(s_shuffled, s_shuffled) / np.sum(s_shuffled)

    # Step 5: Build Laplacian
    L_rand = np.diag(np.sum(W_rand, axis=1)) - W_rand

    # Step 6: Eigen decomposition of Laplacian
    eigenvalues, eigenvectors = la.eigh(L_rand)
    idx = np.argsort(eigenvalues)
    eigenvectors_sorted = eigenvectors[:, idx]
    eigenvectors_flipped = np.fliplr(eigenvectors_sorted)

    return eigenvectors_flipped


def compute_sur_signal(n_brain_regions,eigenvectors,graph_signal):
    
    # create vector to randomize phase
    random_signs = np.round(np.random.rand(n_brain_regions,))
    random_signs[random_signs == 0] = -1
    random_signs = np.diag(random_signs)

    # create surrogate signal using random harmonics and phase randomization
    gft_coefficients        = np.matmul(np.transpose(eigenvectors), graph_signal) 
    eigvector_surr          = np.matmul(eigenvectors, random_signs)
    surrogate_signal        = np.matmul(eigvector_surr, gft_coefficients)

    return surrogate_signal

def comp_SDI_sur(eigenvectors_flip,surrogate_signal,LFF,HFF,n_brain_regions):

    #filtering and inverse GFT of surrogate signal on OBS harmonics and filters!        
    gft_coefficients    = np.matmul(np.transpose(eigenvectors_flip),surrogate_signal)
    low_freq_component  = np.matmul(LFF, gft_coefficients)
    high_freq_component = np.matmul(HFF, gft_coefficients)

    coupling_norm   = np.zeros(n_brain_regions) 
    decoupling_norm = np.zeros(n_brain_regions)
    for j in range(n_brain_regions):
        coupling_norm[j]  = np.linalg.norm(low_freq_component[j,:]) # norm of the projected series in low graph frequency 
        decoupling_norm[j]= np.linalg.norm(high_freq_component[j,:]) # norm of the projected series in high graph frequency       
    
    SDI_sur = np.divide(decoupling_norm,coupling_norm) 
    
    return SDI_sur
