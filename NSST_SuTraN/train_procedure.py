import torch
import torch.nn as nn

from tqdm import tqdm
import os
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

# Importing functionality for Uniform Case-Based Sampling (UCBS) 
# (part of CaLenDiR training)
from CaLenDiR_Utils.case_based_sampling import sample_train_instances, precompute_indices

from NSST_SuTraN.train_utils import NSST_Loss
from NSST_SuTraN.inference_procedure import inference_loop

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train_epoch(model,
                training_loader, 
                scalar_task, 
                out_mask, 
                optimizer, 
                loss_fn, 
                epoch_number, 
                max_norm):
    """Train the Non-Sequential, Single-Task (NSST) Encoder-Only variant 
    of SuTraN on the specified `scalar_task` for the current epoch.

    Parameters
    ----------
    model : SuTraN_NSST
        The initialized and current version of a single-task, 
        encoder-only version of SuTraN for scalar single-task PPM, 
        predicting solely remaining runtime, binary or multi-class 
        outcome respectively. 
    training_loader : torch.utils.data.DataLoader
        Loader yielding NSST batches (prefix tensors plus the scalar label).
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
    optimizer : torch optimizer
        torch.optim.AdamW optimizer. Should already be initialized and 
        wrapped around the parameters of `model`. 
    loss_fn : NSST_Loss
        Task-specific loss wrapper that handles masking for outcome tasks.
    epoch_number : int
        Current epoch index, used strictly for logging/tqdm context.
    max_norm : float, optional
        Maximum norm for gradient clipping, by default 2.
    """

    if scalar_task=='remaining_runtime': 
        loss_string = 'MAE'
    
    elif scalar_task=='binary_outcome': 
        loss_string = 'BCE'
    
    elif scalar_task=='multiclass_outcome': 
        loss_string = 'CCE'

    # Tracking average loss over all batches 
    running_loss = []

    # Tracking original gradient norm and ultimate gradient norm after 
    # clipping at max_norm. 
    original_norm_glb = []
    clipped_norm_glb = []

    # initializing two auxiliary counters accounting for skipped non-valid batches 
    # (possible exception handling invalid outcome batch)
    num_batches_processed = 0 
    num_batches_skipped = 0 

    # Iterating over batches 
    for batch_num, data in tqdm(enumerate(training_loader), desc="Batch calculation at epoch {}.".format(epoch_number)):
        
        # out_mask can only be True if outcome_bool=True
        if out_mask: 
            inputs = data[:-2] 
            labels = data[-2] # tensor of shape (batch_size,1) or (batch_size,)
            instance_mask_out = data[-1] # torch.bool and shape (batch_size,)
            instance_mask_out = instance_mask_out.to(device)

            # Exception handling: skipping batch if there is not at least 
            # one valid outcome instance present in this batch 
            numeric_mask = (instance_mask_out==False).to(torch.float32) # (batch_size,)

            if torch.sum(numeric_mask) == 0:
                print("Exception Handling triggered: Invalid outcome batch encountered and skipped.")
                num_batches_skipped += 1 
                continue 

        else:
            inputs = data[:-1]
            # labels: tensor of shape (batch_size, window_size, 1), 
            # (batch_size,1) or (batch_size,) for RRT, BO or MCO
            labels = data[-1]
            instance_mask_out = None 


        num_batches_processed += 1 

        # Sending inputs and labels to GPU
        inputs = [input_tensor.to(device) for input_tensor in inputs]
        labels = labels.to(device)

        # Restoring gradients to 0 for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss 
        loss = loss_fn(outputs, labels, instance_mask_out)

        # Compute gradients 
        loss.backward()

        # Keep track of original gradient norm 
        original_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        original_norm_glb.append(original_norm.item())

        # Clip gradient norm
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        clipped_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        clipped_norm_glb.append(clipped_norm.item())

        # Adjust learning weights
        optimizer.step()

        # Tracking losses and metrics
        running_loss.append(loss.item())

    print("=======================================")
    print("End of epoch {}".format(epoch_number))
    print("=======================================")
    num_batches = len(running_loss)

    average_global_loss_epoch = sum(running_loss) / num_batches
    print("Average {} loss {} prediction over all batches this epoch: {}".format(loss_string, scalar_task, average_global_loss_epoch))

    average_og_norm = sum(original_norm_glb)/num_batches
    average_clipped_norm = sum(clipped_norm_glb)/num_batches
    print("Average original gradient norm: {} this epoch".format(average_og_norm))
    print("Average clipped gradient norm: {} this epoch".format(average_clipped_norm))

    epoch_averages = average_global_loss_epoch, average_og_norm, average_clipped_norm

    return model, optimizer, epoch_averages


def train_model(model, 
                optimizer, 
                train_dataset, 
                val_dataset, 
                start_epoch, 
                num_epochs, 
                scalar_task,
                path_name,
                batch_size, 
                clen_dis_ref, 
                og_caseint_train, 
                og_caseint_val,
                median_caselen,
                mean_std_rrt=None, 
                out_mask=False, 
                instance_mask_out_train=None, 
                instance_mask_out_val=None, 
                num_outclasses=None,
                patience=24, 
                lr_scheduler_present=False, 
                lr_scheduler=None, 
                best_MAE_rrt=1e9, 
                best_BCE=1e9, 
                best_auc_pr=-1,
                best_auc_roc=-1,
                best_CE_MCO=1e9, 
                best_macro_F1=-1, 
                best_weighted_F1=-1, 
                max_norm=2., 
                seed=None):
    """Train the Non-Sequential, Single-Task (NSST) Encoder-Only variant.

    Parameters
    ----------
    model : SuTraN_NSST
        The initialized and current version of a single-task, 
        encoder-only version of SuTraN for scalar single-task PPM, 
        predicting solely remaining runtime, binary or multi-class 
        outcome respectively. 
    optimizer : torch optimizer
        torch.optim.AdamW optimizer. Should already be initialized and 
        wrapped around the parameters of `model`. 
    train_dataset : tuple of torch.Tensor
        Contains the tensors comprising the training set. This 
        dataset should already be tailored towards the single-task, 
        encoder-only SuTraN version for scalar prediction, and hence 
        only contain the prefix tensors and one label tensor, either the 
        remaining runtime or outcome labels, depending on the 
        `scalar_task` parameter. 
    val_dataset : tuple of torch.Tensor
        Contains the tensors comprising the validation set. This 
        dataset should already be tailored towards the single-task, 
        encoder-only SuTraN version for scalar prediction, and hence 
        only contain the prefix tensors and one label tensor, either the 
        remaining runtime or outcome labels, depending on the 
        `scalar_task` parameter. 
    start_epoch : int
        Number of the epoch from which the training loop is started. 
        First call to ``train_model()`` should be done with 
        ``start_epoch=0``.
    num_epochs : int
        Number of epochs to train. When resuming training with another 
        loop of num_epochs, for the new ``train_model()``, the new 
        ``start_epoch`` argument should be equal to the current one 
        plus the current value for ``num_epochs``.
    scalar_task : {'remaining_runtime', 'binary_outcome', 'multiclass_outcome'}
        The scalar prediction task trained and evaluated for. Either 
        `'remaining_runtime'` `'binary_outcome'` or 
        `'multiclass_outcome'`.
    path_name : str 
        Needed for saving results and callbacks in the 
        appropriate subfolders. This is the path name 
        of the subfolder for which all the results 
        and callbacks (model copies) should be 
        stored for the current event log and 
        model configuration.
    batch_size : int 
        Batch size used during training. 
    clen_dis_ref : bool 
        If `True`, Case Length Distribution-Reflective (CaLenDiR) 
        Training is performed. This includes the application of Uniform 
        Case-Based Sampling (UCBS) of instances each epoch, and 
        Suffix-Length-Normalized Loss Functions. If `False`, the default 
        training procedure, in which all instances are used for training 
        each epoch and in which no loss function normalization is 
        performed (and hence in which case-length distortion is not 
        addressed), is performed. 
    og_caseint_train : torch.Tensor 
        Tensor of dtype torch.int64 and shape 
        `(N_train,)`. Contains the integer-mapped case IDs of the 
        original training set cases from which each of the `N_train` 
        instances have been derived. Used for Uniform Case-Based Sampling 
        (UCBS) in case CaLenDiR training is adopted. 
    og_caseint_val : torch.Tensor 
        Tensor of dtype torch.int64 and shape 
        `(N_val,)`. Contains the integer-mapped case IDs of the 
        original validation set cases from which each of the `N_val` 
        instances have been derived. Used for computing the CaLenDiR 
        (weighted) metrics instead of the instance-based metrics if 
        `clen_dis_ref=True`. These metrics are used for early stopping 
        and final callback selection. 
    median_caselen : int
        Median number of events per original case in the training log. 
        Used by the CaLenDiR sampling routine to anchor 
        case-length-aware subsampling.
    mean_std_rrt : {list of float, None}, optional
        List consisting of two floats, the training mean and standard 
        deviation of the remaining runtime labels (in seconds). Needed 
        for de-standardizing remaining runtime predictions and labels, 
        such that the MAE can be expressed in seconds (and minutes). 
        Note, only required in case `scalar_task='remaining_runtime'`.
        Otherwise, the default of `None` should be retained.
        Mean is the first entry, std the second.
    out_mask : bool
        Whether an instance-level outcome mask is applied to discard
        instances whose inputs (i.e. one of its' prefix events) leak the
        outcome label. When `True`, outcome labels/predictions are subset
        before computing loss and metrics.
        The `out_mask` parameter is only considered if 
        `scalar_task` is set to `'binary_outcome'` or 
        `'multiclass_outcome'`, and ignored otherwise. 
    instance_mask_out_train : {torch.Tensor, None}, optional
        Tensor of dtype torch.bool and shape 
        `(N_train,)`. Containing bool outcome prediction mask, 
        evaluating to True if an instance 
        (/ prefix-suffix pair / prefix) should be masked from 
        contributing to the outcome loss or to the outcome 
        prediction evaluation metrics because of the outcome label 
        being derived from event information already contained within 
        the prefix events (that serve as inputs) to the multi-task 
        model. By default `None`. If no outcome prediction 
        is required (`scalar_task='remaining_runtime'`), or in case outcome 
        prediction is required, but the outcome labels are not derived 
        directly from information contained in other event data, and 
        hence not contained in part of the `N_train` training instances' 
        prefixes (model inputs), the default of `None` should be retained. 
    instance_mask_out_val : {torch.Tensor, None}, optional
        Tensor of dtype torch.bool and shape 
        `(N_val,)`. Containing bool outcome prediction mask, 
        evaluating to True if an instance 
        (/ prefix-suffix pair / prefix) should be masked from 
        contributing to the outcome loss or to the outcome 
        prediction evaluation metrics because of the outcome label 
        being derived from event information already contained within 
        the prefix events (that serve as inputs) to the multi-task 
        model. By default `None`. If no outcome prediction 
        is required (`scalar_task='remaining_runtime'`), or in case outcome 
        prediction is required, but the outcome labels are not derived 
        directly from information contained in other event data, and 
        hence not contained in part of the `N_val` validation instances' 
        prefixes (model inputs), the default of `None` should be retained. 
    num_outclasses : int or None, optional
        The number of outcome classes in case 
        `scalar_task='multiclass_outcome'`. By default `None`. 
    patience : int, optional. 
        Max number of epochs without any improvement in any of the 
        validation metrics. After `patience` epochs without any 
        improvement, the training loop is terminated early. By default 24.
    lr_scheduler_present : bool, optional
        Indicates whether we work with a learning rate scheduler wrapped 
        around the optimizer. If True, learning rate scheduler 
        included. If False (default), not. 
    lr_scheduler : torch lr_scheduler or None
        If ``lr_scheduler_present=True``, a lr_scheduler that is wrapped 
        around the optimizer should be provided as well. For SuTraN, 
        the ExponentialLR() scheduler should be used. 
    best_MAE_rrt : float, optional
        Best validation Mean Absolute Error for the remaining runtime 
        prediction so far. The defaults apply if the training 
        loop is initialized for the first time for a given configuration. 
        If the training loop is resumed from a certain checkpoint, the 
        best results of the previous training loop should be given. If 
        `scalar_task!='remaining_runtime'`, the defaults can be retained even 
        when resuming the training loop from a checkpoint. 
    best_BCE : float, optional
        Best validation Binary Cross Entropy loss so far. The defaults apply if the training 
        loop is initialized for the first time for a given configuration. 
        If the training loop is resumed from a certain checkpoint, the 
        best results of the previous training loop should be given. If 
        `scalar_task!='binary_outcome'`, the defaults can be retained even 
        when resuming the training loop from a checkpoint. 
    best_auc_pr : float, optional
        Best validation Area Under the Curve for the Precision-Recall curve so far. The defaults apply if the training 
        loop is initialized for the first time for a given configuration. 
        If the training loop is resumed from a certain checkpoint, the 
        best results of the previous training loop should be given. If 
        `scalar_task!='binary_outcome'`, the defaults can be retained even 
        when resuming the training loop from a checkpoint. 
    best_auc_roc : float, optional
        Best validation Area Under the ROC Curve observed so far for the 
        outcome task. Initialize to the default when starting from 
        scratch; provide the stored value when resuming training. If 
        `scalar_task!='binary_outcome'`, the defaults can be retained even
        when resuming the training loop from a checkpoint.
    best_CE_MCO : float, optional
        Best validation cross-entropy for multiclass outcome prediction 
        so far. Defaults apply on a fresh run; supply the prior best when 
        resuming. If `scalar_task!='multiclass_outcome'`, the defaults can be
        retained even when resuming the training loop from a checkpoint.
    best_macro_F1 : float, optional
        Best validation macro-averaged F1 score for multiclass outcome 
        prediction. Use the default for new runs and the saved score when 
        resuming. If `scalar_task!='multiclass_outcome'`, the defaults can be
        retained even when resuming the training loop from a checkpoint.
    best_weighted_F1 : float, optional
        Best validation class-weighted F1 score for multiclass outcome 
        prediction. Defaults on fresh runs; restore the previous best 
        when continuing training. If `scalar_task!='multiclass_outcome'`,
        the defaults can be retained even when resuming the training loop
        from a checkpoint.
    max_norm : float, optional
        Maximum norm for gradient clipping, by default 2.
    seed : int or None, optional
        Seed for reproducibility. By default None. When an integer seed 
        is provided, it is used for shuffling and sampling the training
        instances each epoch. If None, the epoch numbers are used as
        seeds for shuffling and sampling.
    """
    outcome_bool = (scalar_task=='binary_outcome') or (scalar_task=='multiclass_outcome')
    # binary out bool
    bin_outbool = (scalar_task=='binary_outcome')
    # multiclass out bool
    multic_outbool = (scalar_task=='multiclass_outcome')
    if outcome_bool and out_mask:
        if instance_mask_out_train is None or not isinstance(instance_mask_out_train, torch.Tensor):
            raise ValueError(
                "When `scalar_task='binary_outcome'` or "
                "`scalar_task='multiclass_outcome'`, and `out_mask` "
                "is True, "
                "'instance_mask_out_train' must be a torch.Tensor and not None."
            )
    if multic_outbool: 
        if num_outclasses is None: 
            raise ValueError(
                "When `scalar_task='multiclass_outcome'`, "
                "'num_outclasses' should be given an integer argument."
            )
            
    # Checking whether GPU is being used
    print("Device: {}".format(device))

    if clen_dis_ref:
        print("CaLenDiR training activated")
    else:
        print("Default training mode. CaLenDiR training not activated.")

    # Adding an additional final tensor to the 
    # train_dataset tensor if needed 
    if outcome_bool and out_mask:
        train_dataset += (instance_mask_out_train, )

    else:
        # Safety mechanism. If no outcome_bool, out_mask should never 
        # be True 
        out_mask = False 

    # in case of default training, DataLoader can be specified over all 
    # instances once. 
    if not clen_dis_ref:
        train_tens_dataset = TensorDataset(*train_dataset)
    else:
        # Prepcompute dictionary with the unique case ID integers as keys 
        # and a tensor containing the integer indices of the instances 
        # derived from each unique case, within the training dataset, 
        # as values. 
        print("Precomputing dictionary mapping each unique ID to its corresponding indices in training dataset.")
        id_to_indices = precompute_indices(og_caseint_train)
    
    # Tracking average loss each consecutive epoch & tracking average 
    # validation metrics computed on validation set after each epoch
    #   - train loss 
    train_losses = []

    # Tracking average og gradient norms (before clipping if needed)
    average_og_norms = []

    # Tracking average final gradient norm (after clipping), used for 
    # updating 
    average_clipped_norms = []

    #   - validation metrics 
    #     - in case of remaining runtime prediction
    if not outcome_bool:
        avg_MAE_stand_RRT_glob, avg_MAE_minutes_RRT_glob = [], []
    #     - in case of binary outcome prediction
    elif bin_outbool: 
        avg_BCE_out_glob, auc_roc_glob, auc_pr_glob= [], [], []
        acc_glob, f1_glob, precision_glob, recall_glob, balanced_accuracy_glob = [], [], [], [], []

    #     - in case of multi-class outcome prediction
    elif multic_outbool: 

        avg_CE_MCO_glob, acc_glob, macro_f1_glob, weighted_f1_glob = [], [], [], []

        macro_precision_glob, weighted_precision_glob, macro_recall_glob, weighted_recall_glob = [], [], [], []
    
    # Initializing Loss Function 
    loss_fn = NSST_Loss(scalar_task=scalar_task, 
                        out_mask=out_mask)
    
    num_epochs_not_improved = 0 

    for epoch in range(start_epoch, start_epoch+num_epochs):
        print(" ")
        print("------------------------------------")
        print('EPOCH {}:'.format(epoch))
        print("____________________________________")

        if seed is not None:
            # Setting seed for reproducable shuffling each epoch
            # epoch+1 to avoid seed_value evaluating to 0 for the first 
            # epoch for all seeds. First epoch matters. 
            seed_value = (epoch+1) * seed 
        else:
            seed_value = epoch

        if clen_dis_ref:
            # CaLenDiR training - UCB Sampling
            print("UCB Sampling...")
            train_sample = sample_train_instances(train_dataset, 
                                                  median_caselen, 
                                                  seed_value, 
                                                  id_to_indices)
            # Creating TensorDataset for the training set 
            train_tens_dataset = TensorDataset(*train_sample)

            # Setting seed for reproducable shuffling each epoch
            torch.manual_seed(seed_value) 
            train_dataloader = DataLoader(train_tens_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        
        else: 
            # Setting seed for reproducable shuffling each epoch
            torch.manual_seed(seed_value) 
            train_dataloader = DataLoader(train_tens_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        # Activate gradient tracking
        model.train(True)
        model, optimizer, epoch_averages = train_epoch(model,
                                                       train_dataloader, 
                                                       scalar_task, 
                                                       out_mask, 
                                                       optimizer, 
                                                       loss_fn, 
                                                       epoch, 
                                                       max_norm)

        
        # Storing tracked metrics 
        train_losses.append(epoch_averages[0])

        average_og_norms.append(epoch_averages[1])

        average_clipped_norms.append(epoch_averages[-1])


        # Set the model to evaluation mode and disabling dropout
        model.eval()


        if clen_dis_ref:
            # Second tuple element contains list of CB metrics
            _, inf_results = inference_loop(model=model, 
                                            inference_dataset=val_dataset,
                                            scalar_task=scalar_task, 
                                            out_mask=out_mask,
                                            mean_std_rrt=mean_std_rrt, 
                                            og_caseint=og_caseint_val,
                                            instance_mask_out=instance_mask_out_val,
                                            num_outclasses=num_outclasses, 
                                            results_path=None, 
                                            val_batch_size=4096)
        else: 
            # First tuple contains list of IB (default) metrics 
            inf_results, _ = inference_loop(model=model, 
                                            inference_dataset=val_dataset,
                                            scalar_task=scalar_task, 
                                            out_mask=out_mask,
                                            mean_std_rrt=mean_std_rrt, 
                                            og_caseint=og_caseint_val,
                                            instance_mask_out=instance_mask_out_val,
                                            num_outclasses=num_outclasses, 
                                            results_path=None, 
                                            val_batch_size=4096)
        
        print("-----------------")
        print("Validation set metrics NSST {}:".format(scalar_task))
        print("-----------------")

        better = False
        
        # Storing average validation metrics 
        if not outcome_bool: # RRT prediction
            avg_MAE_stand_RRT, avg_MAE_minutes_RRT = inf_results
            avg_MAE_stand_RRT_glob.append(avg_MAE_stand_RRT)
            avg_MAE_minutes_RRT_glob.append(avg_MAE_minutes_RRT)
            print("Avg MAE RRT prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_stand_RRT, avg_MAE_minutes_RRT))
            if avg_MAE_stand_RRT < best_MAE_rrt: 
                better = True 
                best_MAE_rrt = avg_MAE_stand_RRT
            
            if better==False: 
                num_epochs_not_improved += 1 
            else: 
                num_epochs_not_improved = 0 

        elif bin_outbool: 
            avg_BCE_out, auc_roc, auc_pr, binary_dict = inf_results

            avg_BCE_out_glob.append(avg_BCE_out)

            auc_roc_glob.append(auc_roc)

            auc_pr_glob.append(auc_pr)

            acc = binary_dict['accuracy']
            acc_glob.append(acc)

            f1 = binary_dict['f1']
            f1_glob.append(f1)

            precision = binary_dict['precision']
            precision_glob.append(precision)

            recall = binary_dict['recall']
            recall_glob.append(recall)

            balanced_accuracy = binary_dict['balanced_accuracy']
            balanced_accuracy_glob.append(balanced_accuracy)

            print("Avg BCE outcome prediction validation set: {}".format(avg_BCE_out))
            print("AUC-ROC outcome prediction validation set: {}".format(auc_roc))
            print("AUC-PR outcome prediction validation set: {}".format(auc_pr))
            print("Accuracy outcome prediction validation set: {}".format(acc))
            print("F1 score outcome prediction validation set: {}".format(f1))
            print("Precision outcome prediction validation set: {}".format(precision))
            print("Recall outcome prediction validation set: {}".format(recall))
            print("Balanced Accuracy outcome prediction validation set: {}".format(balanced_accuracy))

            if avg_BCE_out < best_BCE: 
                better = True 
                best_BCE = avg_BCE_out
            
            if auc_pr > best_auc_pr: 
                better = True 
                best_auc_pr = auc_pr 
            
            if auc_roc > best_auc_roc:
                better = True 
                best_auc_roc = auc_roc 

            if better==False:
                num_epochs_not_improved += 1 
            else: 
                num_epochs_not_improved = 0 

        elif multic_outbool: 

            avg_CE_MCO, mc_dict = inf_results

            avg_CE_MCO_glob.append(avg_CE_MCO)

            acc = mc_dict['accuracy']
            acc_glob.append(acc)

            macro_f1 = mc_dict['macro_f1']
            macro_f1_glob.append(macro_f1)

            weighted_f1 = mc_dict['weighted_f1']
            weighted_f1_glob.append(weighted_f1)

            macro_precision = mc_dict['macro_precision']
            macro_precision_glob.append(macro_precision)

            weighted_precision = mc_dict['weighted_precision']
            weighted_precision_glob.append(weighted_precision)

            macro_recall = mc_dict['macro_recall']
            macro_recall_glob.append(macro_recall)

            weighted_recall = mc_dict['weighted_recall']
            weighted_recall_glob.append(weighted_recall)

            print("Avg CE Multi-Class Outcome (MCO) prediction validation set: {}".format(avg_CE_MCO))
            print("Accuracy MCO prediction validation set: {}".format(acc))
            print("Macro F1 MCO prediction validation set: {}".format(macro_f1))
            print("Weighted F1 MCO prediction validation set: {}".format(weighted_f1))
            print("Macro Precision MCO prediction validation set: {}".format(macro_precision))
            print("Weighted Precision MCO prediction validation set: {}".format(weighted_precision))
            print("Macro Recall MCO prediction validation set: {}".format(macro_recall))
            print("Weighted Recall MCO prediction validation set: {}".format(weighted_recall))

            if avg_CE_MCO < best_CE_MCO: 
                better = True 
                best_CE_MCO = avg_CE_MCO
            
            if macro_f1 > best_macro_F1: 
                better = True 
                best_macro_F1 = macro_f1
            
            if weighted_f1 > best_weighted_F1: 
                better = True 
                best_weighted_F1 = weighted_f1
            
            if better==False:
                num_epochs_not_improved += 1 
            else: 
                num_epochs_not_improved = 0 
        

        # Saving checkpoint every epoch 
        model_path = os.path.join(path_name, 'model_epoch_{}.pt'.format(epoch))
        checkpoint = {'epoch:' : epoch, 
                      'model_state_dict': model.state_dict(), 
                      'optimizer_state_dict': optimizer.state_dict()}
        torch.save(checkpoint, model_path)

        if lr_scheduler_present:
            # Update the learning rate
            lr_scheduler.step()

        if num_epochs_not_improved >= patience:
            print("No improvements in primary validation metrics for {} consecutive epochs. Final epoch: {}".format(patience, epoch))
            break
        
    # Writing training progress to csv at the end of the current training loop
    results_path = os.path.join(path_name, 'backup_results.csv')
    epoch_list = [i for i in range(len(train_losses))]

    # Writing training progress to csv at the end of the current training loop
    if not outcome_bool: # RRT prediction
        results = pd.DataFrame(data = {'epoch' : epoch_list, 
                                       'training loss' : train_losses, 
                                       'Average OG gradient norms' : average_og_norms, 
                                       'Average clipped gradient norms' : average_clipped_norms, 
                                       'RRT - standardized MAE validation': avg_MAE_stand_RRT_glob, 
                                       'RRT - mintues MAE validation': avg_MAE_minutes_RRT_glob
                                       })
        results.to_csv(results_path, index=False)
    
    elif bin_outbool:
        results = pd.DataFrame(data = {'epoch' : epoch_list, 
                                       'training loss' : train_losses, 
                                       'Average OG gradient norms' : average_og_norms, 
                                       'Average clipped gradient norms' : average_clipped_norms, 
                                       'Binary Outcome - BCE validation' : avg_BCE_out_glob, 
                                       'Binary Outcome - AUC-ROC validation' : auc_roc_glob, 
                                       'Binary Outcome - AUC-PR validation' : auc_pr_glob, 
                                       'Binary Outcome - Accuracy' : acc_glob, 
                                       'Binary Outcome - F1 score' : f1_glob, 
                                       'Binary Outcome - Precision' : precision_glob, 
                                       'Binary Outcome - Recall' : recall_glob, 
                                       'Binary Outcome - Balanced Accuracy' : balanced_accuracy_glob
                                       })
        results.to_csv(results_path, index=False)

    elif multic_outbool:
        results = pd.DataFrame(data = {'epoch' : epoch_list, 
                                       'training loss' : train_losses, 
                                       'Average OG gradient norms' : average_og_norms, 
                                       'Average clipped gradient norms' : average_clipped_norms, 
                                       'Multi-Class Outcome - CE validation' : avg_CE_MCO_glob, 
                                       'Multi-Class Outcome - Accuracy' : acc_glob, 
                                       'Multi-Class Outcome - Macro-F1 score' : macro_f1_glob, 
                                       'Multi-Class Outcome - Weighted-F1 score' : weighted_f1_glob, 
                                       'Multi-Class Outcome - Macro-Precision' : macro_precision_glob, 
                                       'Multi-Class Outcome - Weighted-Precision' : weighted_precision_glob, 
                                       'Multi-Class Outcome - Macro-Recall' : macro_recall_glob, 
                                       'Multi-Class Outcome - Weighted-Recall' : weighted_recall_glob
                                       })
        results.to_csv(results_path, index=False)








def load_checkpoint(model, path_to_checkpoint, train_or_eval, lr):
    """
    Loads an already-trained model into memory with the learned weights, 
    as well as the optimizer in its state when the model was saved.

    https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html 

    Parameters
    ----------
    model : nn.Module 
        The model instance you want to load the weights into. It should 
        be initialized with the same architecture/hyperparameters as 
        when it was saved.
    path_to_checkpoint : str
        Full path where the checkpoint (.pt file) is stored on disk.
    train_or_eval : {'train', 'eval'}
        Whether you want to resume training ('train') with the 
        loaded model, or only use it for evaluation ('eval'). 
        The returned `model` will be set to the appropriate mode.
    lr : float 
        Learning rate for re-initializing the optimizer, typically 
        the same LR you had at the time of saving. 

    Returns
    -------
    model : nn.Module
        Model instance with loaded weights.
    optimizer : torch.optim.Optimizer
        Optimizer instance with loaded state.
    final_epoch_trained : int
        The last epoch the loaded model was trained at.
        If you want to resume training from this epoch, use:
           start_epoch = final_epoch_trained + 1
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if train_or_eval not in ('train', 'eval'):
        print("ERROR: 'train_or_eval' argument must be either 'train' or 'eval'.")
        return -1, -1, -1

    # Load the checkpoint
    checkpoint = torch.load(path_to_checkpoint, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Initialize a new optimizer, then load its state
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load the epoch at which training ended
    final_epoch_trained = checkpoint["epoch"]

    # Set the model to train or eval mode
    if train_or_eval == "train":
        model.train()
    else:
        model.eval()

    return model, optimizer, final_epoch_trained









            
        

    
    
    
