import torch

def subset_data(dataset, num_categoricals_pref, seq_task):
    """
    Subset the original dataset tuple for the Sequential, Single-Task 
    (SST) variants of SuTraN, such that the labels pertaining to 
    prediction tasks other than the Sequential prediction task (specified
    for the `seq_task` parameter) are removed.

    Parameters
    ----------
    dataset : tuple of torch.Tensor
        Contains the tensors comprising the train, validation or test 
        dataset, including the labels for all prediction heads. 
    num_categoricals_pref : int
        The number of categorical features (including the activity label) 
        contained within each prefix event token.
    seq_task : {'activity_suffix', 'timestamp_suffix'}
        The (sole) sequential prediction task trained and evaluated 
        for. 

    Returns
    -------
    tuple of torch.Tensor
        The subset of tensors from the original dataset tuple, containing 
        only the tensors relevant to the specified sequential prediction 
        task.
    """
    # Retain the first num_categoricals_pref+4 tensors, containing 
    # the prefix event token features, padding mask, and suffix 
    # event token features.
    subset = list(dataset[:num_categoricals_pref + 4])

    # Depending on the seq task, add the appropriate label tensor.
    if seq_task == 'activity_suffix':
        subset.append(dataset[num_categoricals_pref + 6])
    elif seq_task == 'timestamp_suffix':
        subset.append(dataset[num_categoricals_pref + 4])

    return tuple(subset)