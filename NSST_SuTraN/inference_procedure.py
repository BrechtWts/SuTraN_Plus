"""Run batched inference for the encoder-only NSST SuTraN variants.

Handles remaining-runtime, binary outcome, and multiclass outcome heads by
looping over the inference dataloader, aggregating predictions on CPU, and
computing both instance-based and CaLenDiR case-based metrics in one pass.
"""
import torch
from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader
import os

from CaLenDiR_Utils.weighted_metrics_utils import get_weight_tensor, compute_corrected_avg, suflen_normalized_ttne_mae

from NSST_SuTraN.inference_utils import compute_outcome_BCE, compute_AUC_CaseBased, compute_AUC, compute_rrt_results
from NSST_SuTraN.inference_utils import compute_binary_metrics, compute_binary_metrics_CaseBased
from NSST_SuTraN.inference_utils import compute_multiclass_metrics, compute_multiclass_metrics_CaseBased, compute_MCO_CE

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference_loop(model, 
                   inference_dataset,
                   scalar_task, 
                   out_mask, 
                   mean_std_rrt, 
                   og_caseint, 
                   instance_mask_out, 
                   num_outclasses, 
                   results_path=None, 
                   val_batch_size=8192):
    """Inference loop, both for validition set and ultimate test set. 
    Caters to all, Non-Sequantial, Single-Task (NSST) variants of SuTraN, 
    which comprises the following prediction tasks: 

    #. Remaining Runtime (RRT) prediction
    #. Binary Outcome (BO) prediction
    #. Multi-Class Outcome (MCO) prediction

    Parameters
    ----------
    model : SuTraN_NSST
        The initialized and current version of a single-task, 
        encoder-only version of SuTraN for scalar single-task PPM, 
        predicting solely remaining runtime or outcome 
        respectively. 
    inference_dataset : tuple of torch.Tensor
        Contains the tensors comprising the inference dataset. This 
        dataset should already be tailored towards the single-task, 
        encoder-only SuTraN version for scalar prediction, and hence 
        only contain the prefix tensors and one label tensor, either the 
        remaining runtime or outcome labels, depending on the 
        `scalar_task` parameter, using the
        `NSST_SuTraN.convert_tensordata.subset_data` function.
    scalar_task : {'remaining_runtime', 'binary_outcome', 'multiclass_outcome'}
        The scalar prediction task trained and evaluated for. Either 
        `'remaining_runtime'` `'binary_outcome'` or 
        `'multiclass_outcome'`.
    out_mask : bool
        Whether an instance-level outcome mask is applied to discard
        instances whose inputs (i.e. one of its' prefix events) leak the
        outcome label. When `True`, outcome labels/predictions are subset
        before computing loss and metrics.
        The `out_mask` parameter is only considered if 
        `scalar_task` is set to `'binary_outcome'` or 
        `'multiclass_outcome'`, and ignored otherwise. 
    mean_std_rrt : list of float or None
        List consisting of two floats, the training mean and standard 
        deviation of the remaining runtime labels (in seconds). Needed 
        for de-standardizing remaining runtime predictions and labels, 
        such that the MAE can be expressed in seconds (and minutes). 
        Ignored when `scalar_task` is set to `'binary_outcome'` or 
        `'multiclass_outcome'`.
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
        model. If no outcome prediction
        is required (`scalar_task='remaining_runtime'`), or in case outcome
        prediction is required, but the outcome labels are not derived
        directly from information contained in other event data, and 
        hence not contained in part of the `N_val` validation instances' 
        prefixes (model inputs), it should be set to `None`.
    num_outclasses : int or None 
        The number of outcome classes in case 
        `scalar_task='multiclass_outcome'`. Otherwise `None`. 
    results_path : None or str, optional
        The absolute path name of the folder in which the final 
        evaluation results should be stored. The default of None should 
        be retained for intermediate validation set computations.
    val_batch_size : int, optional
        Batch size for iterating over inference dataset. By default 8192. 
      
    Returns
    -------
    tuple
        `(ib_metrics, cb_metrics)` where the contents depend on
        `scalar_task`:
        - `remaining_runtime`: `[MAE_stand, MAE_minutes]` plus the
          CaLenDiR-weighted counterparts.
        - `binary_outcome`: `[BCE, AUC_ROC, AUC_PR, metric_dict]` and the
          matching case-based set.
        - `multiclass_outcome`: `[CE, metric_dict]` and the CaLenDiR
          equivalents.

    Notes
    -----
    Additional explanations commonly referred tensor dimensionalities: 

    * `num_prefs` : the integer number of instances, aka
        prefix-suffix pairs, contained within the inference dataset. Also
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
    # Verification step
    outcome_bool = (scalar_task=='binary_outcome') or (scalar_task=='multiclass_outcome')
    # binary out bool
    bin_outbool = (scalar_task=='binary_outcome')
    # multiclass out bool
    multic_outbool = (scalar_task=='multiclass_outcome')
    if outcome_bool and out_mask:
        if instance_mask_out is None or not isinstance(instance_mask_out, torch.Tensor):
            raise ValueError(
                "When `scalar_task='binary_outcome'` or "
                "`scalar_task='multiclass_outcome'`, and `out_mask` "
                "is True, "
                "'instance_mask_out' must be a torch.Tensor and not None."
            )
    if multic_outbool: 
        if num_outclasses is None: 
            raise ValueError(
                "When `scalar_task='multiclass_outcome'`, "
                "'num_outclasses' should be given an integer argument."
            )
    # Creating TensorDataset and corresponding DataLoader out of 
    # `inference_dataset`. 
    inf_tensordataset = TensorDataset(*inference_dataset)
    inference_dataloader = DataLoader(inf_tensordataset, batch_size=val_batch_size, shuffle=False, drop_last=False, pin_memory=True)


    # Retrieving label tensor 
    # Shape: 
    # - (num_prefs, window_size, 1) if remaining runtime prediction 
    # - (num_prefs, 1) if binary out prediction
    # - (num_prefs,) if multic out prediction
    labels_global = inference_dataset[-1] 


    if not outcome_bool: # remaining runtime labels
        # labels_global = inference_dataset[-1] # shape (num_prefs, window_size, 1)
        # Extracting only first remaining runtime label for each instance, 
        # pertaining to the ground-truth remaining runtime from the last 
        # observed prefix event 
        labels_global = labels_global[:, 0, 0] # (num_prefs, )

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
        # Shape (num_prefs_out,1) in case of binary out pred 
        # or (num_prefs_out,) in case of multic out pred
        labels_global = labels_global[retain_bool_out] 
    

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        # Initializing global tensor for storing model outputs on CPU
        # Shape (number of instances, ) after inference loop

        if not multic_outbool: 
            # for binary outcome and RRT prediction 
            preds_global = torch.tensor(data=[], dtype=torch.float32)
        
        else: 
            # for multiclass prediction we need outputs of shape
            # (num_prefs, num_outclasses)
            preds_global = torch.empty((0, num_outclasses), dtype=torch.float32)

        # Iterating over the inference batches
        for valbatch_num, vdata in tqdm(enumerate(inference_dataloader), desc="Validation batch calculation"):
                vinputs = vdata[:-1]
                # Assign all input tensors to GPU
                vinputs = [vinput_tensor.to(device) for vinput_tensor in vinputs]

                # Prediction
                # shape (batch_size, 1) in case of RRT and BO prediction,
                # shape (batch_size, num_outclasses) in case of MCO
                # prediction
                voutputs = model(vinputs) 

                if not multic_outbool: 
                    # Preds of shape # (B, 1) = (batch_size, 1)
                    # in case of RRT or Binary Outcome (BO) prediction
                    # are to be reshaped to shape (B,1)
                    voutputs = voutputs[:, 0] # (B,)
                

                # Storing predictions temporarily 
                # Concats correctly for all prediction tasks 
                preds_global = torch.cat((preds_global, voutputs.cpu()), dim=0) 
        
        if outcome_bool and out_mask: 
            # Subsetting relevant tensors for masking 'leaking' inference 
            # instances for outcome prediction for metric computation 
            # shape (num_prefs_out,) if bin_outbool, 
            # shape (num_prefs_out, num_outclasses) if multic_outbool
            preds_global = preds_global[retain_bool_out].clone() 

        # Computing batch inference metrics.
        # Get weight tensor for each instance such that each original inference 
        # set case contributes equally, regardless of the original case's sequence 
        # length / total number of events. 
        # 
        # - weights : torch.Tensor of shape (num_prefs,) and dtype torch.float32 
        # - num_cases : integer, denoting the original number of cases in the inference 
        #               set from which the 'num_prefs' instances have been derived. 

        if not out_mask: 
            weights, num_cases = get_weight_tensor(og_caseint=og_caseint)
        
        elif outcome_bool and out_mask:
            weights, num_cases = get_weight_tensor(og_caseint=og_caseint_out)

        # remaining runtime metrics 
        if not outcome_bool:
            MAE_rrt_stand, MAE_rrt_seconds = compute_rrt_results(rrt_pred=preds_global, 
                                                                 rrt_labels=labels_global, 
                                                                 mean_std_rrt=mean_std_rrt) # (num_prefs,)
            
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

            # instance-based MAE metrics 
            avg_MAE_stand_RRT = (torch.sum(MAE_rrt_stand) / MAE_rrt_stand.shape[0]).item() # Scalar 
            avg_MAE_seconds_RRT = (torch.sum(MAE_rrt_seconds) / MAE_rrt_seconds.shape[0]).item() # Scalar 
            avg_MAE_minutes_RRT = avg_MAE_seconds_RRT / 60 # Scalar

        # outcome metrics outcome prediction
        elif bin_outbool: 
            # compute BCE score for each of the instances. Either a tensor of shape
            # (num_prefs,), or (num_prefs_out,) in case of out_mask.
            inference_BCE = compute_outcome_BCE(out_pred=preds_global, out_labels=labels_global) 

            # Compute average CB BCE 
            avg_BCE_out_CB = compute_corrected_avg(metric_tens=inference_BCE, 
                                                    weight_tens=weights, 
                                                    num_cases=num_cases)
            
            # CaLenDiR's Case-Based AUC-ROC and AUC-PR computations
            auc_roc_CB, auc_pr_CB = compute_AUC_CaseBased(out_pred=preds_global, 
                                                          out_labels=labels_global,
                                                          sample_weight=weights)
            
            
            # Compute scalar CB metrics based on a default 0.5 threshold.
            binary_CB_dict = compute_binary_metrics_CaseBased(out_pred_prob=preds_global, 
                                                              out_labels=labels_global, 
                                                              sample_weight=weights)

            # Compute Instance-Based (IB) metrics

            avg_BCE_out = (torch.sum(inference_BCE) / inference_BCE.shape[0]).item()

            #   AUC-ROC and AUC-PR computations
            auc_roc, auc_pr = compute_AUC(out_pred=preds_global, out_labels=labels_global)

            #   Compute scalar IB metrics based on a treshold 
            binary_IB_dict = compute_binary_metrics(out_pred_prob=preds_global, 
                                                    out_labels=labels_global)
            



        
        elif multic_outbool: 
            # compute CE loss for each of the instances. Either a tensor of shape 
            # (num_prefs,), or (num_prefs_out,) in case of out_mask. 
            inference_CE = compute_MCO_CE(out_pred=preds_global, out_labels=labels_global) 

            # Compute Case-Based (CB) MCO prediction metrics 

            #   Compute average CB CE MCO 
            avg_CE_MCO_CB = compute_corrected_avg(metric_tens=inference_CE, 
                                                  weight_tens=weights, 
                                                  num_cases=num_cases)
            
            #   Compute remaining CB MCO inference metrics 
            mc_CB_dict = compute_multiclass_metrics_CaseBased(out_pred_global=preds_global, 
                                                              out_labels=labels_global, 
                                                              sample_weight=weights)
            
            # Compute Instance-Based (IB) MCO prediction metrics 

            #   Compute average IB CE MCO 
            avg_CE_MCO = (torch.sum(inference_CE) / inference_CE.shape[0]).item()

            #   Compute remaining IB MCO inference metrics 
            mc_IB_dict = compute_multiclass_metrics(out_pred_global=preds_global, 
                                                    out_labels=labels_global)


    # Write predictions for final test set inference to disk if specified 
    if results_path: 
        os.makedirs(results_path, exist_ok=True)
        preds_string = scalar_task + '_preds.pt' 
        preds_path = os.path.join(results_path, preds_string)
        torch.save(preds_global, preds_path)

    if bin_outbool:
        return_list_CB = [avg_BCE_out_CB, auc_roc_CB, auc_pr_CB, binary_CB_dict]
        return_list_IB = [avg_BCE_out, auc_roc, auc_pr, binary_IB_dict]
    
    elif not outcome_bool: 
        return_list_CB = [avg_MAE_stand_RRT_CB, avg_MAE_minutes_RRT_CB]
        return_list_IB = [avg_MAE_stand_RRT, avg_MAE_minutes_RRT]
    
    elif multic_outbool: 
        return_list_CB = [avg_CE_MCO_CB, mc_CB_dict]
        return_list_IB = [avg_CE_MCO, mc_IB_dict]

    return return_list_IB, return_list_CB


        
