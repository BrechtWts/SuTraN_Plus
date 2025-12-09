"""
This module contains the functionality for 
Uniform Case-Based Sampling (UCBS). 
"""

import torch 
import numpy as np
from collections import defaultdict


# Function to precompute indices per unique ID
def precompute_indices(int_ID):
    """Compute dictionary with unique case ID integers as keys, and 
    tensors containing the integer indices of the instances derived 
    from each unique case, within the training dataset, as values. 

    Parameters
    ----------
    int_ID : torch.Tensor
        Dtype torch.int64 and shape `(N,)`, with `N` being the number of 
        training instances, and hence the size of the outermost dimension 
        of all tensors contained within `data_tuple`. 

    Returns
    -------
    id_to_indices : dict
        Mapping from each unique case identifier to a tensor of row
        indices within the training dataset where that case appears.
    """
    unique_ids = int_ID.unique()
    id_to_indices = {uid.item(): (int_ID == uid).nonzero(as_tuple=True)[0] for uid in unique_ids}
    return id_to_indices

# Function to sample m instances per unique ID
def sample_train_instances(data_tuple, m, seed_int, id_to_indices):
    """Sample, for each of the `T` original training set cases (from 
    which `N` instances have been derived) at random `m` instances. If a 
    certain case has a length smaller than `m`, and hence the amount of 
    instances derived from that case is smaller than `m`, `m` instances 
    are sampled with replacement. For cases for which more than `m` 
    instances have been derived, sampling is done without replacement. 

    By doing so, each case contributes an equal amount of training 
    instances, avoiding an artificial overrepresentation of cases with 
    larger case lengths (i.e. larger number of events), and hence 
    avoiding case length distortion in the training set. 

    This function should be called again at the start of each subsequent 
    epoch, with a different integer value for the `seed_int` parameter, 
    such that a slightly different subset is returned for each epoch, 
    thereby also enhancing regularization. 

    Parameters
    ----------
    data_tuple : tuple of torch.Tensor
        Tuple containing the tensors comprising the training set. This 
        includes, i.a., the labels. All tensors have an outermost 
        dimension of the same size, i.e. `N`, the number of 
        original training set instances / prefix-suffix pairs. 
    m : int
        Median case length, computed over the original cases from 
        which training instances are derived. 
    seed_int : int
        Integer based on which the seeds will be set to ensure 
        reproducability and, for each epoch, identical sampling for all 
        benchmark models. 
    id_to_indices : dict
        Precomputed dictionary mapping each unique ID to its 
        corresponding indices in int_ID.

    Returns
    -------
    sampled_data_tuple : tuple of torch.Tensor
        Tuple containing the tensors comprising a subset of the 
        training set instances. This includes, i.a., the labels. 
        All tensors have an outermost dimension of the same size, i.e. 
        `T*m`, with `T` the number of original training set cases from 
        which the `N` original training set instances have been derived, 
        and `m` the median case length. 

    Notes
    -----
    The following notation is used within the comment lines accompanying 
    the code:

    * `T` : the number of original training set cases parsed into 
      multiple instances. 
    
    * `N` : the number of training set instances derived from the `T` 
      training set cases. 

    """
    # Ensure reproducibility with numpy and torch
    torch.manual_seed(seed_int)
    np.random.seed(seed_int)
    
    sampled_indices = []

    for uid, indices in id_to_indices.items():
        indices = indices.numpy()  # Convert to numpy array for sampling

        # Sample with replacement for cases with a number of events 
        # smaller than m (i.e. with less than m instances derived).
        if len(indices) < m:
            sampled_indices.extend(np.random.choice(indices, m, replace=True))
        
        # Sample without replacement, at random, otherwise
        else:
            sampled_indices.extend(np.random.choice(indices, m, replace=False))
    
    sampled_indices = torch.tensor(sampled_indices, dtype=torch.int64)
    
    sampled_data_tuple = tuple(tensor[sampled_indices] for tensor in data_tuple)

    return sampled_data_tuple