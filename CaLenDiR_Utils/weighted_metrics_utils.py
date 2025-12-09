import pandas as pd 
import numpy as np 
import torch 


def get_weight_tensor(og_caseint):
    """Create weight tensor of dtype torch.float32 and shape 
    (num_instances,), with num_instances being the amount of test 
    set instances contained within the test or validation set. Each 
    original case i assigned to the test set is parsed into `n_i` 
    instances / prefix-suffix pairs, with `n_i` being the number of 
    events of case i. As such, (extremely) lengthy cases would influence 
    the average evaluation metrics (significantly) more compared to the 
    cases with a more moderate sequence length. To counter this 
    artificial bias in the 
    results, the score contributions of each instance is weighted by 
    the inverse of the total amount of instances derived from the 
    original case from which the instances have been derived. 

    The adjusted average is computed by the weighted sum of evaluation 
    scores over all instaneces, divided by the total amount of original 
    test set cases from which those instances have been derived. 

    Parameters
    ----------
    og_caseint : torch.Tensor
        Tensor of dtype torch.int64 and shape (num_instances,), 
        containing for each of the 'num_instances' test or validation set 
        instances (/ prefix-suffix pairs) the integer-mapped 
        case ID of the original test or validation set case from which 
        the instance was derived. 
    
    Notes
    -----
    The following notation is used within the comment lines accompanying 
    the code:

    * `T` : the number of original test or validation set cases parsed 
      into multiple instances. 
    
    * `N` : the number of test or validation set instances derived from 
      the `T` test or validation set cases. 
    """
    # Calculate the number of occurrences for each element in the tensor
    # Both tensors are of shape (T,). 
    #   - unique_elements: tensor containing the unique integer case IDs 
    #   - counts: tensor containing the number of instances pertaining 
    #     to each original integer case ID
    unique_elements, counts = og_caseint.unique(return_counts=True)


    # Dictionary with integer case IDs as the keys and the corresponding 
    # number of instances as values 
    count_dict = dict(zip(unique_elements.tolist(), counts.tolist()))

    # Compute the weight tensor of shape (N, )
    weights = torch.tensor([1.0 / count_dict[val.item()] for val in og_caseint], dtype=torch.float32) # (N,)

    # Compute the number of original cases from which test instances have 
    # been derived
    num_cases = unique_elements.shape[0] # integer 
    
    return weights, num_cases

def compute_corrected_avg(metric_tens, 
                          weight_tens,
                          num_cases):
    """Compute the corrected weighted average. Each original case i 
    is assigned to the inference set is parsed into `n_i` instances / 
    prefix-suffix pairs, with `n_i` being the number of events of case i. 
    As such, (extremely) lengthy cases would influence the average 
    evaluation metrics (significantly) more compared to the cases with a 
    more moderate sequence length. To counter this artificial bias in the 
    results, the score contributions of each instance is weighted by 
    the inverse of the total amount of instances derived from the 
    original case from which the instances have been derived. 

    The adjusted average is computed by the weighted sum of evaluation 
    scores over all instances, divided by the total amount of original 
    inference set cases from which those instances have been derived. 

    The inference set can be either the final test set, or the validation 
    set processed after every epoch. 

    The `metric_tens` parameter contains the individual scores for 
    each of the `N` inference set instances. This can be either the 
    Damerau-Levenshtein Similarity (DLS) for activity suffix prediction, 
    or the Mean Absolute Error (MAE) for remaining runtime prediction. To 
    be able to compute the corrected averages for the MAE of timestamp 
    suffix prediction, since the majority of instances contribute 
    multiple prediction points and hence multiple MAE values (equal to 
    the suffix length), rather than a scalar metric, an intermediate 
    Suffix-Length normalization, in which the averages per instance 
    are computed first, is required. After having done that, and hence 
    having obtained a scalar metric for each instance, one can obtain 
    the final corrected averages for timestamp suffix prediction using 
    this function as well. 

    Finally, `metric_tens` can also be the Suffix-Length-Normalized 
    validation loss for the CRTP-LSTM benchmark, and the one-step-ahead 
    validation loss for the SEP-LSTM benchmark. 

    Parameters
    ----------
    metric_tens : torch.Tensor
        Tensor of dtype torch.float32 and shape (N,) containing the 
        evaluation score (DLS or MAE) for each of the N test set 
        instances. 
    weight_tens : torch.Tensor
        Tensor of dtype torch.float32 and shape (N,) containing the 
        corrected weight 
    num_cases : int
        The number of original test set cases (`T`) from which the `N` 
        instances have been derived. Serving as the denominator of 
        the weighted sum to arrive at the final corrected average 
        evaluation scores. 

    Notes
    -----
    The following notation is used within the comment lines accompanying 
    the code:

    * `T` : the number of original test set cases parsed into multiple 
      instances. (Equal to `num_cases`.)
    
    * `N` : the number of test set instances derived from the `T` test 
      set cases.

    """
    corrected_avg = torch.sum(metric_tens * weight_tens).item() / num_cases

    return corrected_avg # scalar 

def suflen_normalized_ttne_mae(MAE_ttne, suf_len):
    """Compute the average MAE per instance for timestamp suffix 
    prediction, only taking into account the actual non-padded 
    suffix events (in the ground-truth suffixes). 

    Parameters
    ----------
    MAE_ttne : torch.Tensor
        Dtype torch.float32 and shape (batch_size, window_size). 
        Contains the Mean Absolute Errors (MAE) for all 'window_size' 
        timestamp suffix predictions, for all of the 'batch_size' 
        instances. This still includes the invalid MAEs pertaining 
        to (right-padded) padding events. 
    suf_len : torch.Tensor
        Dtype torch.int64 and shape (batch_size,). Contains the 
        ground-truth suffix length for each of the 'batch_size' 
        instances. For each of the 'batch_size' instances, only 
        the MAE values up to the corresponding index in `suf_len` minus 
        1 should be taken into account. 

    Returns
    -------
    MAE_ttne_instance : torch.Tensor 
        Dtype torch.float32 and shape (batch_size,). Contains, for each 
        of the 'batch_size' instances, the mean MAE for timestamp suffix 
        prediction, averaged over the ground-truth actual (non-padded) 
        events for each instance. 
    """
    batch_size = MAE_ttne.shape[0]
    window_size = MAE_ttne.shape[1]

    # Creating mask to exclude the padding events 
    #       Generate a tensor with values counting from 0 to window_size
    counting_tensor = torch.arange(window_size, dtype=torch.int64) # (window_size,)
    #       Repeat the tensor along the first dimension to match the desired shape
    counting_tensor = counting_tensor.unsqueeze(0).repeat(batch_size, 1) # (batch_size, window_size)
    #       Boolean tensor 
    before_end_token = counting_tensor <= (suf_len-1).unsqueeze(-1) # (batch_size, window_size)

    # Masking the MAEs for the padded events 
    masked_MAE = MAE_ttne*before_end_token.to(torch.float32) # (batch_size, window_size)

    # Computing average MAE per instance 
    summed_MAE = torch.sum(masked_MAE, dim=-1) # (batch_size,)
    avg_MAE = summed_MAE / suf_len # (batch_size,)

    return avg_MAE