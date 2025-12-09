import torch



def add_prefix_CLStoken(dataset, 
                        cardinality_categoricals_pref,  
                        num_numericals_pref):
    """Add a CLS token to all prefix tensors. This function should be 
    called outside of the training function, which requires the already 
    CLS augmented dataset in case CLS token prediction is used for the 
    single-task scalar prediction models. 

    The CLS token is added to the beginning of the (right-padded) prefix 
    event token sequence. 

    These CLS tokens are represented in the following manner among the 
    different categorical and numerical prefix event token features: 

    #. All prefix categoricals receive an additional level, representing 
       the CLS token. 

    #. The prefix numerics are given the value of zero for the CLS 
       leading CLS token. Given the standardization of features, 
       zero corresponds to the mean value of each feature. This 
       is a neutral value, adding little information. 
    
    #. The padding mask is appended with an additional leading False, 
       such that the attention mechanisms will effectively attend to 
       the leading CLS  token as well. 

    Parameters
    ----------
    dataset : tuple of torch.Tensor
        Tuple containing the tensors comprising the training, validation, 
        or test set. This includes, i.a., the labels. All tensors have an 
        outermost dimension of the same size, i.e. `N_train`, the number 
        of original train, val or test instances / prefix-suffix pairs. 
        They also share the second dimension (dim=1), being equal to 
        window_size, the (padded) sequence length. 
    cardinality_categoricals_pref : list of int
        List of `num_categoricals` integers. Each integer entry 
        i (i = 0, ..., `num_categoricals`-1) contains the cardinality 
        of the i'th categorical feature of the encoder prefix events. 
        The order of the cardinalities should match the order in 
        which the categoricals are fed as inputs. Note that for each 
        categorical, an extra category should be included to account 
        for missing values. I.e. these numbers do not include the 
        added padding level at index 0. 
    num_numericals_pref : int 
        Number of numerical features of the prefix events. 
    """
    num_categoricals_pref = len(cardinality_categoricals_pref)

    # converting dataset to list to make it mutable
    dataset = list(dataset)

    # Selecting padding_mask_input 
    padding_mask_input = dataset[num_categoricals_pref+1] # (num_prefs, window_size)

    num_prefs = padding_mask_input.shape[0]

    # Adding a CLS token in front of the prefix event sequence for each categorical 
    #   Deriving index of CLS level for each categorical. Index equal to cardinality + 1,
    #   since cardinality does not account for the padding level at index 0.
    cls_level_list = [car+1 for car in cardinality_categoricals_pref]

    #   Iterating over the tensors containing the prefix event token's 
    #   categorical features. These are the `num_categoricals_pref` first 
    #   tensors in the `dataset` list. 
    for i in range(num_categoricals_pref):
        cat_tens = dataset[i] # (batch_size, window_size)
        cls_index = cls_level_list[i]

        # creating cls token tensor for every instance 
        # shape (num_prefs, 1)
        cls_cat = torch.full(size=(num_prefs, 1), fill_value=cls_index, dtype=torch.int64)

        # Adding for each of the num_prefs instances the CLS token as the first 
        # token of the sequence of prefix event tokens 
        cat_tens = torch.cat(tensors=(cls_cat, cat_tens), dim=-1) # (num_prefs, window_size+1)

        # Updating the categorical tensor within the dataset 
        dataset[i] = cat_tens 


    # Augmenting the (single) tensor containing all the numeric features 
    # pertaining to the sequence of prefix event tokens 
    num_ftrs_pref = dataset[num_categoricals_pref] # shape (num_prefs, window_size, num_numericals_pref)

    # creating CLS token tensor for all numerics 
    cls_numerics = torch.full(size=(num_prefs, 1, num_numericals_pref), fill_value=0., dtype=torch.float32)

    # Adding in front of current numerics, shape (num_prefs, window_size+1, num_numericals_pref)
    num_ftrs_pref = torch.cat(tensors=(cls_numerics, num_ftrs_pref), dim=1) 

    # Updating corresponding tensor in `dataset`
    dataset[num_categoricals_pref] = num_ftrs_pref


    # Augmenting the padding mask 
    cls_pad = torch.zeros(size=(num_prefs,1), dtype=torch.int64).to(torch.bool) # (num_prefs,1)
    padding_mask_input = torch.cat(tensors=(cls_pad, padding_mask_input), dim=-1) # (num_prefs, window_size+1)

    dataset[num_categoricals_pref+1] = padding_mask_input

    return tuple(dataset) 

def subset_data(dataset, num_categoricals_pref, scalar_task):
    """
    Subset the original dataset tuple for the Non-Sequential, Single-Task 
    (NSST) variants of SuTraN, such that only the categorical and 
    numerical prefix event token features, the padding mask, and the 
    tensor containing the labels pertaining to the Non-Sequential 
    prediction task (specified for the `scalar_task` parameter) are 
    retained. 

    The tensors at the following indices are contained: 

    The tensors at indices 0 to `num_categoricals_pref-1`, each 
    containing a categorical prefix event token feature. The tensor at 
    index `num_categoricals_pref`, containing all numerical prefix event 
    token features. Lastly, depending on `scalar_task`, the tensor 
    containing the labels for that task is retained. 
    Parameters
    ----------
    dataset : tuple of torch.Tensor
        Contains the tensors comprising the train, validation or test 
        dataset, including the labels for all prediction heads. 
    num_categoricals_pref : int
        The number of categorical features (including the activity label) 
        contained within each prefix event token.
    scalar_task : {'remaining_runtime', 'binary_outcome', 'multiclass_outcome'}
        The scalar prediction task trained and evaluated for. Either 
        `'remaining_runtime'` `'binary_outcome'` or 
        `'multiclass_outcome'`. 
        If:
            - 'remaining_runtime': also retains the label tensor at index 
              num_categoricals_pref + 5.
            - 'binary_outcome' or 'multiclass_outcome': retains the label 
              tensor at index num_categoricals_pref + 7.

    Returns
    -------
    tuple: 
        A tuple of torch.Tensors containing the subset of tensors 
        required for the specified task.
        - Each tensor from index 0 to `num_categoricals_pref-1` contains 
          one of the `num_categoricals_pref` categorical prefix event 
          token features. The categorical tensor at index 
          `num_categoricals_pref-1` contains the activity labels of the 
          prefix event tokens. 
        - The tensor at index `num_categoricals_pref` contains all 
          numerical prefix event features. Dtype torch.float32. 
        - The tensor at index `num_categoricals_pref+1` contains the 
          padding mask, for masking out the prefix padding tokens 
          from contributing to the self-attention and prediction. 
        - The tensor at index `num_categoricals_pref+2`, equivalent to 
          index `-1` contains the labels pertaining to the prediction 
          task specified by the `scalar_task` parameter. 
    """
    if ((scalar_task=='binary_outcome') or (scalar_task=='multiclass_outcome')):
        last_index = len(dataset) - 1 
        # index outcome labels tensor
        out_index = num_categoricals_pref + 7

        if last_index != out_index: 
            raise ValueError(
                "You specified `scalar_task='{}'`, ".format(scalar_task) + 
                "but `dataset` does not contain an outcome label tensor. "
            )
    # Retain the first num_categoricals_pref+2 tensors.
    subset = list(dataset[:num_categoricals_pref + 2])

    # Depending on the scalar task, add the appropriate label tensor.
    if scalar_task == 'remaining_runtime':
        subset.append(dataset[num_categoricals_pref + 5])
    elif scalar_task in ('binary_outcome', 'multiclass_outcome'):
        subset.append(dataset[num_categoricals_pref + 7])
    else:
        raise ValueError(f"Invalid scalar_task '{scalar_task}'. Expected one of "
                         f"'remaining_runtime', 'binary_outcome', or 'multiclass_outcome'.")

    return tuple(subset)
