"""Functionality for conducting parallel inference over batches for the 
Sequential Single-Task SuTraN versions. (Activity Suffix and Timestamp 
Suffix prediction). 
"""


import torch
from tqdm import tqdm
from SST_SuTraN.inference_environment import BatchInference

from torch.utils.data import TensorDataset, DataLoader
import os

from CaLenDiR_Utils.weighted_metrics_utils import get_weight_tensor, compute_corrected_avg, suflen_normalized_ttne_mae

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference_loop(model, 
                   inference_dataset,
                   seq_task,  
                   num_categoricals_pref,
                   mean_std_ttne, 
                   mean_std_tsp, 
                   mean_std_tss,
                   og_caseint, 
                   results_path=None, 
                   val_batch_size=8192):
    """Inference loop, both for validition set and ultimate test set. 
    Caters to both Sequential, Single-Task (SST) variants of SuTraN, 
    which comprises the following prediction tasks: 

    #. Activity Suffix prediction
    #. Timestamp Suffix prediction

    Parameters
    ----------
    model : SuTraN_SST
        The initialized and current version of a Sequential, Single-Task
        (SST) version of SuTraN for sequential single-task PPM, 
        predicting solely the activity suffix or timestamp suffix.
    inference_dataset : tuple of torch.Tensor
        Tensors already subset for the SST task at hand (e.g., via
        `SST_SuTraN.convert_tensordata.subset_data`). The tuple still
        includes the complete prefix stack plus every suffix-event helper
        tensor—activity indices and both time proxy features—while all
        extraneous label tensors are removed except for the label that
        matches the active task. The `SST_SuTraN.SuTraN_SST` forward pass
        then consumes only the suffix tokens it can update
        autoregressively (activity indices for
        `seq_task='activity_suffix'`, or the two time numerics for
        `seq_task='timestamp_suffix'`), ignoring the other suffix tensor
        because that head is not being generated. Consequently, the final
        tensor in the tuple is either of shape `(N, window_size)`,
        comprising the activity targets, or of shape `(N, window_size, 1)`
        comprising the timestamp suffix targets.
    seq_task : {'activity_suffix', 'timestamp_suffix'}
        The (sole) sequential prediction task trained and evaluated 
        for. 
    num_categoricals_pref : int
        The number of categorical features (including the activity label) 
        contained within each prefix event token.
    mean_std_ttne : list of float
        Training mean and standard deviation used to standardize the time 
        till next event (in seconds) target. Needed for re-converting 
        ttne predictions to original scale. Mean is the first entry, 
        std the second. Ignored when `seq_task=='activity_suffix'`.
    mean_std_tsp : list of float
        Training mean and standard deviation used to standardize the time 
        since previous event (in seconds) feature of the decoder suffix 
        tokens. Needed for re-converting time since previous event values 
        to original scale (seconds). Mean is the first entry, std the 2nd.
        Ignored when `seq_task=='activity_suffix'`.
    mean_std_tss : list of float
        Training mean and standard deviation used to standardize the time 
        since start (in seconds) feature of the decoder suffix tokens. 
        Needed for re-converting time since start to original scale 
        (seconds). Mean is the first entry, std the 2nd.
        Ignored when `seq_task=='activity_suffix'`.
    og_caseint : torch.Tensor 
        Tensor of dtype torch.int64 and shape `(N_inf,)`, with `N_inf` 
        the number of instances contained within the inference (test or 
        validation) set. 
        Contains the integer-mapped case IDs of the 
        original inference set cases from which each of the `N_val` 
        instances have been derived. Used for computing the CaLenDiR 
        (weighted) metrics instead of the instance-based metrics. These 
        metrics are used for early stopping and final callback selection. 
    results_path : None or str, optional
        Directory for persisting raw predictions. This argument is only
        honored when the internal `store_preds` flag is enabled (default
        False), so leave it as `None` for standard validation runs.
    val_batch_size : int, optional
        Batch size for iterating over inference dataset. By default 8192. 

    Returns
    -------
    tuple
        `(ib_metrics, cb_metrics, pref_suf_results)` where:
        - `ib_metrics` holds the default instance-based scores
          (`[DLS]` for activities or `[MAE_std, MAE_min]` for timestamps).
        - `cb_metrics` contains the CaLenDiR case-based counterparts.
        - `pref_suf_results` is a pair of dictionaries mapping prefix
          lengths and suffix lengths to `[metric, num_instances]`.
      
    Notes
    -----
    Additional explanations commonly referred tensor dimensionalities:

    * `num_prefs` : the integer number of instances, aka 
        prefix-suffix pairs, contained within the inference dataset 
        for which this `BatchInference` instance is initialized. Also 
        often referred to as `batch_size` in the comment lines 
        complementing the code. 

    * `window_size` : the maximum sequence length of both the prefix 
        event sequences, as well as the generated suffix event 
        predictions. 

    * `num_activities` : the total number of possible activity labels 
        to be predicted. This includes the padding and end token. The 
        padding token will however always be masked, such that it 
        cannot be predicted. Also referred to as `num_classes`. 
    """
    act_sufbool = (seq_task=='activity_suffix') 
    ts_sufbool = (seq_task=='timestamp_suffix')
    if not (act_sufbool or ts_sufbool):
        raise ValueError(
            "`seq_task={}` is not a valid argument. It ".format(seq_task) + 
            "should be either `'activity_suffix'` or "
            "`'timestamp_suffix'`."
        )
    
    # Hardcoded temporarily. Whether or not to store the predictions to disk too
    store_preds = False 

    # Creating TensorDataset and corresponding DataLoader out of 
    # `inference_dataset`. 
    inf_tensordataset = TensorDataset(*inference_dataset)
    inference_dataloader = DataLoader(inf_tensordataset, batch_size=val_batch_size, shuffle=False, drop_last=False, pin_memory=True)


    # Retrieving label tensor 
    # Shape: 
    # - (num_prefs, window_size) if `seq_task=='activity_suffix'`
    # - (num_prefs, window_size, 1) if `seq_task=='timestamp_suffix'`
    labels_global = inference_dataset[-1] 

    window_size = labels_global.shape[1]

    num_prefs = labels_global.shape[0]

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        # Initializing global tensors for storing model outputs on CPU
        # The two directly underneath have shape 
        # (num_prefs, window_size) after inference loop
        if act_sufbool: 
            preds_global = torch.empty((0, window_size), dtype=torch.int64)
        elif ts_sufbool: 
            preds_global = torch.empty((0, window_size), dtype=torch.float32)
            

        # Iterating over the inference batches 
        for valbatch_num, vdata in tqdm(enumerate(inference_dataloader), desc="Validation batch calculation"):
                vinputs = vdata[:-1]
                # Assign all input tensors to GPU
                vinputs = [vinput_tensor.to(device) for vinput_tensor in vinputs]

                # Retrieving activity or timestamp suffix predictions 
                # for current batch. In both cases of shape 
                # (B, W)
                voutputs = model(vinputs, 
                                 window_size, 
                                 mean_std_ttne, 
                                 mean_std_tsp, 
                                 mean_std_tss) 

                # Storing predictions temporarily 
                # Concats correctly for all prediction tasks 
                preds_global = torch.cat((preds_global, voutputs.cpu()), dim=0) 
        

        # Computing batch inference metrics 

        # Get weight tensor for each instance such that each original inference 
        # set case contributes equally, regardless of the original case's sequence 
        # length / total number of events. 
        # 
        # - weights : torch.Tensor of shape (num_prefs,) and dtype torch.float32 
        # - num_cases : integer, denoting the original number of cases in the inference 
        #               set from which the 'num_prefs' instances have been derived. 

        weights, num_cases = get_weight_tensor(og_caseint=og_caseint)

        if results_path and store_preds: 
            os.makedirs(results_path, exist_ok=True)

            # Specifying paths to save the prediction tensors and writing them 
            # to disk. 
            if act_sufbool: 
                #   Activity suffix predictions 
                suffix_acts_decoded_path = os.path.join(results_path, 'suffix_acts_decoded.pt')
                torch.save(preds_global, suffix_acts_decoded_path)
            
            elif ts_sufbool:
                #   Timestamp suffix predictions 
                suffix_ttne_preds_path = os.path.join(results_path, 'suffix_ttne_preds.pt')
                torch.save(preds_global, suffix_ttne_preds_path)
        
        # Initializing Batchinference object for computing inf metrics 
        infer_env = BatchInference(preds=preds_global, 
                                   labels=labels_global, 
                                   seq_task=seq_task, 
                                   mean_std_ttne=mean_std_ttne, 
                                   mean_std_tsp=mean_std_tsp, 
                                   mean_std_tss=mean_std_tss)
        # Retrieving individual validation metric components for each of 
        # the 'num_prefs' instances, for all prediction targets. 

        if ts_sufbool:
            # Compute initial TTNE metrics
            # both of shape (num_prefs, window_size). 
            # Only the MAE values pertaining to the non-padded suffix event  
            # tokens (in the two initial TTNE metrics) should still be   
            # selected before computing global averages (see infra). 
            MAE_ttne_stand, MAE_ttne_seconds = infer_env.compute_ttne_results()

        elif act_sufbool: 
            # (normalized) Damerau-Levenshtein similarity activity suffix prediction
            dam_lev = infer_env.damerau_levenshtein_distance_tensors() # (num_prefs, )
            dam_lev_similarity = 1. - dam_lev # (num_prefs,)

        # Determining the ground-truth suffix length for each of the num_prefs 
        # instances 
        suf_len_global = infer_env.get_actual_length()
        # Incrementing with 1
        suf_len_global = suf_len_global + 1

        # Initializing a global tensor to store the prefix lengths of all inference instances 
        pad_mask_global = inference_dataset[num_categoricals_pref+1] # (num_prefs, window_size)
        pref_len_global = torch.argmax(pad_mask_global.to(torch.int64), dim=-1) # (batch_size,)

        # Prefixes of maximum length (window_size) have no True values in 
        # the padding mask and hence prefix length 0 is (falsely) derived 
        # replacing 0s with window_size 
        pref_len_global = torch.where(pref_len_global == 0, window_size, pref_len_global) # (num_prefs,)


        #############################################
        #       Case-based metric computation       #
        #############################################

        if act_sufbool:
            # Activity suffix 
            # Case-Based (CB) average DLS 
            avg_dam_lev_CB = compute_corrected_avg(metric_tens=dam_lev_similarity, 
                                                weight_tens=weights, 
                                                num_cases=num_cases)
            
        elif ts_sufbool:
            # Timestamp suffix prediction 
            #   Computing average MAE per instance 
            MAE_ttne_stand_CB = suflen_normalized_ttne_mae(MAE_ttne=MAE_ttne_stand.clone(), 
                                                        suf_len=suf_len_global) # (num_prefs,)
            MAE_ttne_seconds_CB = suflen_normalized_ttne_mae(MAE_ttne=MAE_ttne_seconds.clone(), 
                                                            suf_len=suf_len_global) # (num_prefs,)
            
            MAE_ttne_minutes_CB = MAE_ttne_seconds_CB / 60 # (num_prefs,)
            
            #   CB averages TTNE prediction
            #       standardized 
            avg_MAE_ttne_stand_CB = compute_corrected_avg(metric_tens=MAE_ttne_stand_CB, 
                                                        weight_tens=weights, 
                                                        num_cases=num_cases)
            #       seconds 
            avg_MAE_ttne_seconds_CB = compute_corrected_avg(metric_tens=MAE_ttne_seconds_CB, 
                                                            weight_tens=weights, 
                                                            num_cases=num_cases)
            
            #       minutes 
            avg_MAE_ttne_minutes_CB = avg_MAE_ttne_seconds_CB / 60


        #############################################
        # Instance-based (default) metric computation
        #############################################
        if act_sufbool: 
            # Activity suffix 
            #   normalized Damerau Levenshtein similarity Activity Suffix 
            #   prediction 
            # dam_lev_similarity = 1. - dam_lev # (num_prefs,)
            avg_dam_lev = (torch.sum(dam_lev_similarity) / dam_lev_similarity.shape[0]).item() # Scalar

        elif ts_sufbool: 
            # Time Till Next Event (TTNE) suffix 
            #     Retain only MAE contributions pertaining to 
            #     non-padded suffix events
            counting_tensor = torch.arange(window_size, dtype=torch.int64) # (window_size,)
            #       Repeat the tensor along the first dimension to match the desired shape
            counting_tensor = counting_tensor.unsqueeze(0).repeat(num_prefs, 1) # (num_prefs, window_size)
            #       Compute boolean indexing tensor to, for each of the 
            #       'num_prefs' instances, slice out only the absolute 
            #       errors pertaining to actual non-padded suffix events. 
            before_end_token = counting_tensor <= (suf_len_global-1).unsqueeze(-1) # (num_prefs, window_size)

            avg_MAE_ttne_stand = MAE_ttne_stand[before_end_token] # shape (torch.sum(suf_len_global), )
            avg_MAE_ttne_stand = (torch.sum(avg_MAE_ttne_stand) / avg_MAE_ttne_stand.shape[0]).item()

            avg_MAE_ttne_seconds = MAE_ttne_seconds[before_end_token] # shape (torch.sum(suf_len_global), )
            avg_MAE_ttne_seconds = (torch.sum(avg_MAE_ttne_seconds) / avg_MAE_ttne_seconds.shape[0]).item()

            avg_MAE_ttne_minutes = avg_MAE_ttne_seconds / 60


    ################################################
    # Consolidating instance-based (IB) (default) metrics 
    ################################################

    if act_sufbool: 
        return_list_IB = [avg_dam_lev]

    elif ts_sufbool: 
        return_list_IB = [avg_MAE_ttne_stand, avg_MAE_ttne_minutes]

    ################################################
    # Consolidating Case-Based metrics 
    ################################################

    if act_sufbool: 
        return_list_CB = [avg_dam_lev_CB]
    
    elif ts_sufbool:
        return_list_CB = [avg_MAE_ttne_stand_CB, avg_MAE_ttne_minutes_CB]


    ##################################################
    #   Computing average metrics for instances of   #
    #      different prefix and suffix lengths       #
    ##################################################
    
    # Making dictionaries of the results for over both prefix and suff length. 
    results_dict_pref = {}
    for i in range(1, window_size+1):
        bool_idx = pref_len_global==i
        if act_sufbool:
            dam_levs = dam_lev_similarity[bool_idx].clone()
            num_inst = dam_levs.shape[0]
        elif ts_sufbool:
            MAE_ttne_i = MAE_ttne_minutes_CB[bool_idx].clone()
            num_inst = MAE_ttne_i.shape[0]
        if num_inst > 0:
            if act_sufbool: 
                avg_dl = (torch.sum(dam_levs) / num_inst).item()
                results_i = [avg_dl, num_inst]
            elif ts_sufbool:
                avg_mae_ttne = (torch.sum(MAE_ttne_i) / num_inst).item()
                results_i = [avg_mae_ttne, num_inst]
            results_dict_pref[i] = results_i
    results_dict_suf = {}
    for i in range(1, window_size+1):
        bool_idx = suf_len_global==i
        if act_sufbool:
            dam_levs = dam_lev_similarity[bool_idx].clone()
            num_inst = dam_levs.shape[0]
        elif ts_sufbool:
            MAE_ttne_i = MAE_ttne_minutes_CB[bool_idx].clone()
            num_inst = MAE_ttne_i.shape[0]
        if num_inst > 0:
            if act_sufbool:
                avg_dl = (torch.sum(dam_levs) / num_inst).item()
                results_i = [avg_dl, num_inst]
            elif ts_sufbool: 
                avg_mae_ttne = (torch.sum(MAE_ttne_i) / num_inst).item()
                results_i = [avg_mae_ttne, num_inst]
            results_dict_suf[i] = results_i

    pref_suf_results = [results_dict_pref, results_dict_suf]

    return return_list_IB, return_list_CB, pref_suf_results


        
