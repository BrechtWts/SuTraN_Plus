"""
Inference utilities for SuTraN+.

Orchestrates validation/test loops for the SuTraN+ Transformer:
batches the inference dataset, invokes the model's autoregressive
`eval()` forward pass to collect activity suffix, timestamp suffix,
remaining runtime (optional), and outcome (optional) predictions, and
forwards them to
`BatchInference` for metric aggregation (instance-based and case-based).
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from SuTraN.inference_environment import BatchInference

from torch.utils.data import TensorDataset, DataLoader
import os 
import pickle

from CaLenDiR_Utils.weighted_metrics_utils import get_weight_tensor, compute_corrected_avg, suflen_normalized_ttne_mae

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def inference_loop(model, 
                   inference_dataset,
                   remaining_runtime_head, 
                   outcome_bool, 
                   out_mask, 
                   out_type, 
                   num_outclasses,
                   num_categoricals_pref,
                   mean_std_ttne, 
                   mean_std_tsp, 
                   mean_std_tss,
                   mean_std_rrt, 
                   og_caseint, 
                   instance_mask_out, 
                   results_path=None, 
                   val_batch_size=8192):
    """Inference loop, both for validation set and ultimate test set.

    Parameters
    ----------
    model : SuTraN
        The initialized and current version of a SuTraN neural network. 
        Should be set to evaluation mode to trigger the AR decoding loop 
        within SuTraN's forward method. 
    inference_dataset : tuple of torch.Tensor
        Contains the tensors comprising the inference dataset, including 
        the labels for all prediction heads. 
    remaining_runtime_head : bool
        Whether or not the model is also trained to directly predict the 
        remaining runtime given a prefix.
    outcome_bool : bool
        Indicates whether an outcome head (binary or multiclass,
        depending on ``out_type``) was trained and should be evaluated.
        Outcome predictions are generated at the first decoding step
        only and must stay aligned with the preprocessing pipeline
        that produced the labels.
    out_mask : bool
        Whether an instance-level outcome mask is applied to discard
        instances whose inputs (i.e. one of its' prefix events) leak the
        outcome label. When `True`, outcome labels/predictions are subset
        before computing loss and metrics.
    out_type : {None, 'binary_outcome', 'multiclass_outcome'}
        The type of outcome prediction that is being performed in the 
        multi-task setting. Only taken into account of outcome prediction 
        is included in the event log to begin with, and hence if 
        `outcome_bool=True`. If so, `'binary_outcome'` denotes binary 
        outcome (BO) prediction (binary classification), while 
        `'multiclass_outcome'` denotes multi-class outcome (MCO) 
        prediction (Multi-Class classification). 
    num_outclasses : int or None
        The number of outcome classes in case 
        `outcome_bool=True` and `out_type='multiclass_outcome'`. 
    num_categoricals_pref : int
        The number of categorical features (including the activity label) 
        contained within each prefix event token. For the NDA 
        implementation of SuTraN (`SuTraN_no_context`), this parameter 
        should be set to `1`. 
    mean_std_ttne : list of float
        Training mean and standard deviation used to standardize the time 
        till next event (in seconds) target. Needed for re-converting 
        ttne predictions to original scale. Mean is the first entry, 
        std the second.
    mean_std_tsp : list of float
        Training mean and standard deviation used to standardize the time 
        since previous event (in seconds) feature of the decoder suffix 
        tokens. Needed for re-converting time since previous event values 
        to original scale (seconds). Mean is the first entry, std the 2nd.
    mean_std_tss : list of float
        Training mean and standard deviation used to standardize the time 
        since start (in seconds) feature of the decoder suffix tokens. 
        Needed for re-converting time since start to original scale 
        (seconds). Mean is the first entry, std the 2nd. 
    mean_std_rrt : list of float
        List consisting of two floats, the training mean and standard 
        deviation of the remaining runtime labels (in seconds). Needed 
        for de-standardizing remaining runtime predictions and labels, 
        such that the MAE can be expressed in seconds (and minutes). 
    og_caseint : torch.Tensor 
        Tensor of dtype torch.int64 and shape `(N_inf,)`, with `N_inf` 
        the number of instances contained within the inference (test or 
        validation) set. 
        Contains the integer-mapped case IDs of the 
        original inference set cases from which each of the `N_val` 
        instances have been derived. Used for computing the CaLenDiR 
        (weighted) metrics instead of the instance-based metrics. These 
        metrics are used for early stopping and final callback selection. 
    instance_mask_out : {torch.Tensor, None}
        Tensor of dtype torch.bool and shape 
        `(N_val,)`. Containing bool outcome prediction mask, 
        evaluating to True if an instance 
        (/ prefix-suffix pair / prefix) should be masked from 
        contributing to the outcome loss or evaluation metrics because of
        the outcome label being derived from event information already
        contained within the prefix events (that serve as inputs) to the
        multi-task model. By default `None`. If no outcome prediction
        is required (`outcome_bool=False`), or in case outcome
        prediction is required, but the outcome labels are not derived
        directly from information contained in other event data, and 
        hence not contained in part of the `N_val` validation instances' 
        prefixes (model inputs) (`outcome_bool=True` and 
        `out_mask=False`), it should be set to `None`.
    results_path : None or str, optional
        The absolute path name of the folder in which the final 
        evaluation results should be stored. The default of None should 
        be retained for intermediate validation set computations.
    val_batch_size : int, optional
        Batch size for iterating over inference dataset. By default 8192. 

    Returns
    -------
    list
        Instance-based (IB) averages in the order
        `[avg_MAE_ttne_stand, avg_MAE_ttne_minutes, avg_dam_lev, ...]`
        with remaining-runtime and outcome entries appended when the
        corresponding heads are active.
    list
        Case-based (CB) averages in the analogous order.
    list
        `[results_dict_pref, results_dict_suf]`, each mapping prefix
        (or suffix) length to `[avg_dls, avg_rrt_mae_minutes,
        avg_ttne_mae_minutes, count]`.


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
    # hardcoded boolean. Set to True if you want the individual 
    # predictions for all targets to be written to disk when running 
    # inference on the test set. 
    store_preds = False 

    # Verification step
    if outcome_bool and out_mask:
        if instance_mask_out is None or not isinstance(instance_mask_out, torch.Tensor):
            raise ValueError(
                "When 'outcome_bool' and 'out_mask' are both True, "
                "'instance_mask_out' must be a torch.Tensor and not None."
            )

    # binary out bool
    bin_outbool = (out_type=='binary_outcome')
    # multiclass out bool
    multic_outbool = (out_type=='multiclass_outcome')
    
    if outcome_bool:
        if not (out_type == 'binary_outcome' or out_type == 'multiclass_outcome'):
            raise ValueError(
                "When `outcome_bool=True`, `out_type` must be either "
                "'binary_outcome' or 'multiclass_outcome'."
            )

        if multic_outbool: 
            if num_outclasses is None: 
                raise ValueError(
                    "When `outcome_bool=True` and "
                    "`out_type='multiclass_outcome'`, "
                    "'num_outclasses' should be given an integer argument."
                )
            
    else: 
        out_type = None
        bin_outbool = False 
        multic_outbool = False
        num_outclasses = None

    # Creating TensorDataset and corresponding DataLoader out of 
    # `inference_dataset`. 
    inf_tensordataset = TensorDataset(*inference_dataset)
    inference_dataloader = DataLoader(inf_tensordataset, batch_size=val_batch_size, shuffle=False, drop_last=False, pin_memory=True)

    # Define auxiliary booleans specifying SuTraN's multi-task setup 
    only_rrt = (not outcome_bool) & remaining_runtime_head
    only_out = outcome_bool & (not remaining_runtime_head)
    both_not = (not outcome_bool) & (not remaining_runtime_head)
    both = outcome_bool & remaining_runtime_head

    # If the only additional prediction head (on top of activity and ttne 
    # suffix) is remaining time prediction
    if only_rrt:
        # number of prediction targets (and hence labels) to be 
        # simultaneously predicted 
        num_target_tens = 3

        # Index of the ground-truth activity label tensor in the dataset 
        act_label_index = -1 

    # If the only additional prediction head (on top of activity and ttne 
    # suffix) is outcome prediction
    elif only_out:
        # number of prediction targets (and hence labels) to be 
        # simultaneously predicted 
        num_target_tens = 3

        # Index of the ground-truth activity label tensor in the dataset 
        act_label_index = -2

    # If the additional prediction heads (on top of activity and ttne 
    # suffix) comprise both rrt and outcome prediction
    elif both:
        # number of prediction targets (and hence labels) to be 
        # simultaneously predicted 
        num_target_tens = 4

        # Index of the ground-truth activity label tensor in the dataset 
        act_label_index = -2

    # If their are no additional prediction heads (on top of activity and 
    # ttne suffix)
    elif both_not: 
        # number of prediction targets (and hence labels) to be 
        # simultaneously predicted 
        num_target_tens = 2

        # Index of the ground-truth activity label tensor in the dataset 
        act_label_index = -1
    
    # Retrieving labels 
    labels_global = inference_dataset[-num_target_tens:] 

    if outcome_bool and out_mask: 
        # Subsetting relevant tensors for masking 'leaking' inference 
        # instances for outcome prediction for metric computation 

        #   Computing boolean indexing tensor evaluating to True for 
        #   'num_prefs_out' (<= 'num_prefs') non-leaky instances 
        #   (to be retained)
        retain_bool_out = instance_mask_out==False # shape (num_prefs,)

        # Subsetting integer case IDs for valid outcome instances
        og_caseint_out = og_caseint[retain_bool_out].clone() # (num_prefs_out,)

        # Subsetting outcome labels for non-leaky outcome 
        # instances only 
        out_labels = labels_global[-1] # (num_prefs,1) if BO, (num_prefs,) if MCO
        out_labels_subset = out_labels[retain_bool_out].clone() # (num_prefs_out,1) if BO, (num_prefs_out,) if MCO

        # Reassigning subsetted outcome labels to labels tuple 
        labels_global = list(labels_global)
        labels_global[-1] = out_labels_subset # (num_prefs_out,1) if BO, (num_prefs_out,) if MCO
        labels_global = tuple(labels_global)

    
    # Retrieving seq length (`window_size`, also referred to as W) 
    window_size = labels_global[act_label_index].shape[-1]



    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        # Initializing global tensors for storing model outputs on CPU
        # The two directly underneath have shape 
        # (num_prefs, window_size) after inference loop
        suffix_acts_decoded_global = torch.empty((0, window_size), dtype=torch.int64)
        suffix_ttne_preds_global = torch.empty((0, window_size), dtype=torch.float32)

        if remaining_runtime_head:
            # Shape (number of instances, ) after inference loop
            rrt_pred_global = torch.tensor(data=[], dtype=torch.float32)
        
        if outcome_bool:
            if bin_outbool: # Binary Outcome (BO) prediction. 
                # Shape (num_prefs,) or (num_prefs_out,) after inference loop
                out_pred_global = torch.tensor(data=[], dtype=torch.float32)
            elif multic_outbool: # Multi-Class Outcome (MCO) prediction
                # shape (num_prefs, num_outclasses) or (num_prefs_out, num_outclasses)
                # after inference loop 
                out_pred_global = torch.empty((0, num_outclasses), dtype=torch.float32)


        # Initializing a global tensor to store the prefix lengths of all inference instances 
        # One based already
        pad_mask_global = inference_dataset[num_categoricals_pref+1] # (num_prefs, window_size)
        pref_len_global = torch.argmax(pad_mask_global.to(torch.int64), dim=-1) # (batch_size,)

        # Prefixes of maximum length (window_size) have no True values in 
        # the padding mask and hence prefix length 0 is (falsely) derived 
        # replacing 0s with window_size 
        pref_len_global = torch.where(pref_len_global == 0, window_size, pref_len_global) # (num_prefs,)

        act_labels_global = inference_dataset[act_label_index] # (num_prefs, window_size)
        num_classes = torch.max(act_labels_global).item() + 1


        # Total number of test or validation set instances / 
        # prefix-suffix pairs 
        num_prefs = act_labels_global.shape[0]

        # Derive ground-truth suffix length of each instance 
        suf_len_global = torch.argmax((act_labels_global == (num_classes-1)).to(torch.int64), dim=-1) + 1 # (num_prefs,)

        # Iterating over the inference batches 
        for valbatch_num, vdata in tqdm(enumerate(inference_dataloader), desc="Validation batch calculation"):
                vinputs = vdata[:-num_target_tens]
                # Assign all input tensors to GPU
                vinputs = [vinput_tensor.to(device) for vinput_tensor in vinputs]


                # Decoding the batch_size instances. 
                # NOTE that the model should be set in evaluation mode 
                # in order for it to handle the entire (greedy) decoding 
                # process itself. 
                voutputs = model(vinputs, 
                                 window_size, 
                                 mean_std_ttne, 
                                 mean_std_tsp, 
                                 mean_std_tss)
                
                # Retrieving the different outputs and adding them to 
                # their respective global tensors on the CPU

                #   - Greedily decoded activity suffix 
                suffix_acts_decoded = voutputs[0] # (B, W) torch.int64
                suffix_acts_decoded_global = torch.cat((suffix_acts_decoded_global, suffix_acts_decoded.cpu()), dim=0)

                #   - Predicted TTNE suffix (in standardized scale)
                #     Note: only the predictions up until the decoding 
                #     step in which the END token is predicted in 
                #     `suffix_acts_decoded` should be taken into account. 
                suffix_ttne_preds = voutputs[1] # (B, W) torch.float32
                suffix_ttne_preds_global = torch.cat((suffix_ttne_preds_global, suffix_ttne_preds.cpu()), dim=0)

                #   - remaining runtime and / or outcome predictions if 
                #     trained for. 
                if only_rrt: 
                    # - (direct) remaining runtime predictions. 
                    #   Still in standardized scale. 
                    rrt_pred = voutputs[-1] # (B,) torch.float32
                    rrt_pred_global = torch.cat((rrt_pred_global, rrt_pred.cpu()), dim=-1)

                elif only_out:
                    # - BO or MCO prediction
                    out_pred = voutputs[-1] # (B, ) or (B, num_outclasses), torch.float32
                    out_pred_global = torch.cat((out_pred_global, out_pred.cpu()), dim=0)
                
                elif both:
                    # - (direct) remaining runtime predictions. 
                    #   Still in standardized scale. 
                    rrt_pred = voutputs[-2] # (B,) torch.float32
                    rrt_pred_global = torch.cat((rrt_pred_global, rrt_pred.cpu()), dim=-1)

                    # - BO or MCO prediction
                    out_pred = voutputs[-1] # (B, ) or (B, num_outclasses), torch.float32
                    out_pred_global = torch.cat((out_pred_global, out_pred.cpu()), dim=0)

        if outcome_bool and out_mask: 
            # Subsetting relevant tensors for masking 'leaking' inference 
            # instances for outcome prediction for metric computation 
            # shape (num_prefs_out,) if bin_outbool, 
            # shape (num_prefs_out, num_outclasses) if multic_outbool
            out_pred_global_subset = out_pred_global[retain_bool_out].clone() 

        # Consolidating all predictions 
        outputs_global = (suffix_acts_decoded_global, suffix_ttne_preds_global)

        if remaining_runtime_head:
            outputs_global += (rrt_pred_global,)
        
        if outcome_bool:
            if out_mask:
                outputs_global += (out_pred_global_subset,)
            else:
                outputs_global += (out_pred_global, )
        
        # Write away results for final test set inference if specified 
        if results_path and store_preds: # store preds hardcoded in beginning function
            subfolder_path = results_path
            os.makedirs(subfolder_path, exist_ok=True)

            # Specifying paths to save the prediction tensors and writing 
            # them to disk. 

            #   Activity suffix predictions 
            suffix_acts_decoded_path = os.path.join(subfolder_path, 'suffix_acts_decoded.pt')
            torch.save(suffix_acts_decoded_global, suffix_acts_decoded_path)

            #   Timestamp suffix predictions 
            suffix_ttne_preds_path = os.path.join(subfolder_path, 'suffix_ttne_preds.pt')
            torch.save(suffix_ttne_preds_global, suffix_ttne_preds_path)

            if remaining_runtime_head:
                rrt_pred_path = os.path.join(subfolder_path, 'rrt_pred.pt')
                torch.save(rrt_pred_global, rrt_pred_path)
            if outcome_bool:
                out_pred_path = os.path.join(subfolder_path, 'out_pred.pt')
                torch.save(out_pred_global, out_pred_path)
            
            # Prefix length and suffix length 
            pref_len_path = os.path.join(subfolder_path, 'pref_len.pt')
            torch.save(pref_len_global, pref_len_path)

            suf_len_path = os.path.join(subfolder_path, 'suf_len.pt')
            torch.save(suf_len_global, suf_len_path)

            if outcome_bool and out_mask:
                # Saving original outcome predictions and labels 
                # without having discarded the leaky ones for exploratory 
                # purposes
                og_out_preds_path = os.path.join(subfolder_path, 'og_out_pred.pt')
                torch.save(out_pred_global, og_out_preds_path)

        # Initializing BatchInference object for computing 
        # inference metrics 
        infer_env = BatchInference(preds=outputs_global, 
                                   labels=labels_global, 
                                   mean_std_ttne=mean_std_ttne, 
                                   mean_std_tsp=mean_std_tsp, 
                                   mean_std_tss=mean_std_tss, 
                                   mean_std_rrt=mean_std_rrt, 
                                   remaining_runtime_head=remaining_runtime_head, 
                                   outcome_bool=outcome_bool)
        # Retrieving individual validation metric components for each of 
        # the 'num_prefs' instances, for all prediction targets. 

        # Compute initial TTNE metrics
        # both of shape (num_prefs, window_size). 
        # Only the MAE values pertaining to the non-padded suffix event  
        # tokens (in the two initial TTNE metrics) should still be   
        # selected before computing global averages (see infra). 
        MAE_ttne_stand, MAE_ttne_seconds = infer_env.compute_ttne_results()

        # (normalized) Damerau-Levenshtein similarity activity suffix prediction
        dam_lev = infer_env.damerau_levenshtein_distance_tensors() # (num_prefs, )
        dam_lev_similarity = 1. - dam_lev # (num_prefs,)

        # MAE remaining runtime predictions (standardized scale and in seconds)
        if remaining_runtime_head:
            
            MAE_rrt_stand, MAE_rrt_seconds = infer_env.compute_rrt_results() # (num_prefs,)
            
        if outcome_bool:
            if bin_outbool:
                # infer_env automatically given the appropriate outcome labels and 
                # predictions if needed
                inference_BCE = infer_env.compute_outcome_BCE() # (num_prefs,) or (num_prefs_out,)
            
            elif multic_outbool: 
                # compute CE loss for each of the instances. Either a tensor of shape 
                # (num_prefs,), or (num_prefs_out,) in case of out_mask. 
                inference_CE = infer_env.compute_MCO_CE() 


        # Length differences between predicted and ground-truth suffixes. 
        # Omitted for this paper. In case one wants to keep track of these 
        # metrics too, uncomment the line of code underneath. 
        # length_diff, length_diff_too_early, length_diff_too_late, amount_right = infer_env.compute_suf_length_diffs()

        #############################################
        #       Case-based metric computation       #
        #############################################

        # Get weight tensor for each instance such that each original inference 
        # set case contributes equally, regardless of the original case's sequence 
        # length / total number of events. 
        # 
        # - weights : torch.Tensor of shape (num_prefs,) and dtype torch.float32 
        # - num_cases : integer, denoting the original number of cases in the inference 
        #               set from which the 'num_prefs' instances have been derived. 
        weights, num_cases = get_weight_tensor(og_caseint=og_caseint)


        if outcome_bool and out_mask:
            # Computing separate weights and number of og cases for 
            # outcome prediction in case of outcome instances 
            # leaking the label
            # - weights_out : torch.Tensor of shape (num_prefs_out,) and dtype torch.float32 
            # - num_cases_out : integer, denoting the original number of cases in the inference 
            #               set from which the 'num_prefs' instances have been derived, and for 
            #               which at least one instance is not subject to data leakage wrt 
            #               the outcome label.  
            weights_out, num_cases_out = get_weight_tensor(og_caseint=og_caseint_out)

        # Activity suffix 
        # Case-Based (CB) average DLS 
        avg_dam_lev_CB = compute_corrected_avg(metric_tens=dam_lev_similarity, 
                                               weight_tens=weights, 
                                               num_cases=num_cases)
        
        # Remaining Runtime MAE 
        # CB average MAE (both standardized, in seconds and in minutes)
        if remaining_runtime_head:
            # Standardized CB
            avg_MAE_stand_RRT_CB = compute_corrected_avg(metric_tens=MAE_rrt_stand, 
                                                         weight_tens=weights, 
                                                         num_cases=num_cases)
            # Seconds 
            avg_MAE_seconds_RRT_CB = compute_corrected_avg(metric_tens=MAE_rrt_seconds, 
                                                          weight_tens=weights, 
                                                          num_cases=num_cases)
            # Minutes 
            avg_MAE_minutes_RRT_CB = avg_MAE_seconds_RRT_CB / 60 

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


        # outcome prediction 
        if outcome_bool:
            if out_mask: 
                if bin_outbool:
                    avg_BCE_out_CB = compute_corrected_avg(metric_tens=inference_BCE, 
                                                        weight_tens=weights_out, 
                                                        num_cases=num_cases_out)
                    
                    # CaLenDiR's Case-Based AUC-ROC and AUC-PR computations 
                    auc_roc_CB, auc_pr_CB = infer_env.compute_AUC_CaseBased(sample_weight=weights_out)

                    # Compute scalar CB metrics based on a default 0.5 threshold. 
                    binary_CB_dict = infer_env.compute_binary_metrics_CaseBased(sample_weight=weights_out)

                elif multic_outbool:
                    #   Compute average CB CE MCO 
                    avg_CE_MCO_CB = compute_corrected_avg(metric_tens=inference_CE, 
                                                        weight_tens=weights_out, 
                                                        num_cases=num_cases_out)
                    
                    #   Compute remaining CB MCO inference metrics 
                    mc_CB_dict = infer_env.compute_multiclass_metrics_CaseBased(sample_weight=weights_out)



            else:  # no out mask
                if bin_outbool:
                    avg_BCE_out_CB = compute_corrected_avg(metric_tens=inference_BCE, 
                                                        weight_tens=weights, 
                                                        num_cases=num_cases)
                
                    # CaLenDiR's Case-Based AUC-ROC and AUC-PR computations 
                    auc_roc_CB, auc_pr_CB = infer_env.compute_AUC_CaseBased(sample_weight=weights)

                    # Compute scalar CB metrics based on a default 0.5 threshold. 
                    binary_CB_dict = infer_env.compute_binary_metrics_CaseBased(sample_weight=weights)
                
                elif multic_outbool: 
                    #   Compute average CB CE MCO 
                    avg_CE_MCO_CB = compute_corrected_avg(metric_tens=inference_CE, 
                                                        weight_tens=weights, 
                                                        num_cases=num_cases)
                    
                    #   Compute remaining CB MCO inference metrics 
                    mc_CB_dict = infer_env.compute_multiclass_metrics_CaseBased(sample_weight=weights)
                    
    
            



        #############################################
        # Instance-based (default) metric computation
        #############################################

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

        # Activity suffix 
        #   normalized Damerau Levenshtein similarity Activity Suffix 
        #   prediction 
        # dam_lev_similarity = 1. - dam_lev # (num_prefs,)
        avg_dam_lev = (torch.sum(dam_lev_similarity) / dam_lev_similarity.shape[0]).item() # Scalar

        # Remaining Runtime (RRT)
        if remaining_runtime_head:
            avg_MAE_stand_RRT = (torch.sum(MAE_rrt_stand) / MAE_rrt_stand.shape[0]).item() # Scalar 
            avg_MAE_seconds_RRT = (torch.sum(MAE_rrt_seconds) / MAE_rrt_seconds.shape[0]).item() # Scalar 
            avg_MAE_minutes_RRT = avg_MAE_seconds_RRT / 60 # Scalar 
            # Without averaging
            MAE_rrt_minutes = MAE_rrt_seconds / 60 # (num_prefs, )
        
        if results_path and store_preds:
            # Writing the tensors containing the DLS and MAE RRT for each individual 
            # test set instance / test set prefix-suffix pair, to disk. 
            dam_lev_sim_path = os.path.join(subfolder_path, 'dam_lev_similarity.pt')
            torch.save(dam_lev_similarity, dam_lev_sim_path)
            
            if remaining_runtime_head:
                MAE_rrt_minutes_path = os.path.join(subfolder_path, 'MAE_rrt_minutes.pt')
                torch.save(MAE_rrt_minutes, MAE_rrt_minutes_path)

        # outcome 
        if outcome_bool:
            if bin_outbool: # Binary Outcome (BO)
                avg_BCE_out = (torch.sum(inference_BCE) / inference_BCE.shape[0]).item()
                # AUC-ROC and AUC-PR computations
                auc_roc, auc_pr = infer_env.compute_AUC()

                #   Compute scalar IB metrics based on a treshold 
                binary_IB_dict = infer_env.compute_binary_metrics()
            
            elif multic_outbool: # Multi-Class Outcome (MCO)
                #   Compute average IB CE MCO 
                avg_CE_MCO = (torch.sum(inference_CE) / inference_CE.shape[0]).item()

                #   Compute remaining IB MCO inference metrics 
                mc_IB_dict = infer_env.compute_multiclass_metrics()

            

        # # Length differences: 
        # Omitted for this paper. In case one wants to keep track of these 
        # metrics too, uncomment the line of code underneath. 
        # total_num = length_diff.shape[0]
        # num_too_early = length_diff_too_early.shape[0]
        # num_too_late = length_diff_too_late.shape[0]
        # percentage_too_early = num_too_early / total_num
        # percentage_too_late = num_too_late / total_num
        # percentage_correct = amount_right.item() / total_num
        # mean_absolute_length_diff = (torch.sum(torch.abs(length_diff)) / total_num).item()
        # mean_too_early = (torch.sum(torch.abs(length_diff_too_early)) / num_too_early).item()
        # mean_too_late = (torch.sum(torch.abs(length_diff_too_late)) / num_too_late).item()

    ################################################
    # Consolidating instance-based (IB) (default) metrics 
    ################################################

    return_list_IB = [avg_MAE_ttne_stand, avg_MAE_ttne_minutes, avg_dam_lev]

    if remaining_runtime_head:
        return_list_IB += [avg_MAE_stand_RRT, avg_MAE_minutes_RRT]
    if outcome_bool:
        if bin_outbool:
            return_list_IB += [avg_BCE_out, auc_roc, auc_pr, binary_IB_dict]
        elif multic_outbool: 
            return_list_IB += [avg_CE_MCO, mc_IB_dict]
        
            

    ################################################
    # Consolidating Case-Based metrics 
    ################################################

    return_list_CB = [avg_MAE_ttne_stand_CB, avg_MAE_ttne_minutes_CB, avg_dam_lev_CB]

    if remaining_runtime_head:
        return_list_CB += [avg_MAE_stand_RRT_CB, avg_MAE_minutes_RRT_CB]

    if outcome_bool:
        if bin_outbool:
            return_list_CB += [avg_BCE_out_CB, auc_roc_CB, auc_pr_CB, binary_CB_dict]
        
        elif multic_outbool: 
            return_list_CB += [avg_CE_MCO_CB, mc_CB_dict]

    ##################################################
    #   Computing average metrics for instances of   #
    #      different prefix and suffix lengths       #
    ##################################################
    
    # Making dictionaries of the results for over both prefix and suff length. 
    results_dict_pref = {}
    for i in range(1, window_size+1):
        bool_idx = pref_len_global==i
        dam_levs = dam_lev_similarity[bool_idx].clone()
        MAE_rrt_i = MAE_rrt_minutes[bool_idx].clone()
        MAE_ttne_i = MAE_ttne_minutes_CB[bool_idx].clone()
        num_inst = dam_levs.shape[0]
        if num_inst > 0:
            avg_dl = (torch.sum(dam_levs) / num_inst).item()
            avg_mae = (torch.sum(MAE_rrt_i) / num_inst).item()
            avg_mae_ttne = (torch.sum(MAE_ttne_i) / num_inst).item()
            results_i = [avg_dl, avg_mae, avg_mae_ttne, num_inst]
            results_dict_pref[i] = results_i
    results_dict_suf = {}
    for i in range(1, window_size+1):
        bool_idx = suf_len_global==i
        dam_levs = dam_lev_similarity[bool_idx].clone()
        MAE_rrt_i = MAE_rrt_minutes[bool_idx].clone()
        MAE_ttne_i = MAE_ttne_minutes_CB[bool_idx].clone()
        num_inst = dam_levs.shape[0]
        if num_inst > 0:
            avg_dl = (torch.sum(dam_levs) / num_inst).item()
            avg_mae = (torch.sum(MAE_rrt_i) / num_inst).item()
            avg_mae_ttne = (torch.sum(MAE_ttne_i) / num_inst).item()
            results_i = [avg_dl, avg_mae, avg_mae_ttne, num_inst]
            results_dict_suf[i] = results_i

    pref_suf_results = [results_dict_pref, results_dict_suf]

    return return_list_IB, return_list_CB, pref_suf_results