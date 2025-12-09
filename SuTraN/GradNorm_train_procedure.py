"""
Train loop tailored to GradNorm-based multi-task optimization.

This module mirrors the structure of the generic SuTraN training
procedure but augments it with GradNorm bookkeeping to dynamically
scale task losses. It sets up the optimizer and GradNorm-specific
parameters, logs task-wise gradients for monitoring, and exposes
`train_epoch`/`train_model` helpers that wrap the SuTraN model in a
GradNorm-aware update cycle.

References
----------
Chen, Z., Badrinarayanan, V., Lee, C. Y., & Rabinovich, A. (2018).
GradNorm: Gradient normalization for adaptive loss balancing in deep
multitask networks. In ICML (pp. 794-803). PMLR.
"""


import torch
import torch.nn as nn
import numpy as np

from SuTraN.train_utils_GradNorm import MultiOutputLoss_GradNorm
from tqdm import tqdm
import os
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from SuTraN.inference_procedure import inference_loop

# Importing functionality for Uniform Case-Based Sampling (UCBS) 
# (part of CaLenDiR training)
from CaLenDiR_Utils.case_based_sampling import sample_train_instances, precompute_indices

# Import tensorboard SummaryWriter to log several metrics and variables 
from torch.utils.tensorboard import SummaryWriter

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log_variables(writer, 
                  type_string, 
                  component_string_list, 
                  scalar_value_list, 
                  training_iteration):
    """Log variable values using tensorboard. 

    Parameters
    ----------
    writer : torch.utils.tensorboard.writer.SummaryWriter
        An instance of `SummaryWriter` from the PyTorch TensorBoard 
        utility used for logging training metrics.
    type_string : str
        String denoting the type of metric or variable that is 
        being monitored. E.g. "Loss" for denoting the different losses 
        that are being tracked in our multi-task setup. 
    component_string_list : str
        List of strings denoting the various instances of the specific 
        metric / variable pertaining to the type specified by the 
        `type_string` parameter that is being tracked. Order should 
        match with the order in the `scalar_value_list` parameter. 
    scalar_value_list : {list of float, np.array of float, tuple of float} 
        The list of values of the metric / variable instances to be 
        logged. Order should match with the order in the 
        `component_string_list` parameter.
    training_iteration : int
        The training iteration for which we are logging that value. 
        Depending on the use-case, this could refer to the batch 
        iteration number or the epoch number. 
    """
    for i in range(len(component_string_list)):
        component_string = component_string_list[i]
        scalar_value = scalar_value_list[i]
        # Specify data identifier 
        tag_string = '{}/{}'.format(type_string, component_string)
        # Log metric 
        writer.add_scalar(tag_string, scalar_value, training_iteration)



def train_epoch(model, 
                training_loader, 
                remaining_runtime_head, 
                outcome_bool,
                out_mask, 
                out_type,
                model_optim, 
                grad_optim,
                loss_fn, 
                batch_interval,
                epoch_number, 
                warmup_epoch, # bool 
                exponential_transformation, # bool
                writer, 
                writer_bool):
    """
    Run one GradNorm-aware training epoch.

    The routine iterates over the training loader, optionally performs a
    warm-up pass with uniform task weights, and then applies GradNorm to
    jointly update the model parameters and the learnable task weights.
    It also tracks task-wise losses, gradient norms, and (optionally)
    logs them to TensorBoard.

    Parameters
    ----------
    model : SuTraN
        SuTraN model currently in training mode. Forward calls must return
        the tuple of task outputs expected by ``loss_fn``.
    training_loader : torch.utils.data.DataLoader
        Loader that yields batches of prefix/suffix tensors together with
        the aligned targets (and, if applicable, outcome masks).
    remaining_runtime_head : bool
        Whether the model includes the direct remaining-runtime head.
        Determines which target tensors are unpacked from ``data``.
    outcome_bool : bool
        Whether an outcome prediction head is trained (binary or
        multiclass, depending on ``out_type``).
    out_mask : bool
        If ``True``, each batch carries an instance-level outcome mask to
        drop prefixes that already reveal the label.
    out_type : {None, 'binary_outcome', 'multiclass_outcome'}
        Type of outcome head configured on the model. Ignored when
        ``outcome_bool`` is ``False``.
    model_optim : torch.optim.Optimizer
        Optimizer that updates the SuTraN parameters.
    grad_optim : torch.optim.Optimizer
        Optimizer that updates the GradNorm task weights.
    loss_fn : Callable
        GradNorm loss wrapper. Must accept ``initial_step`` and
        ``instance_mask_out`` keywords and return
        ``(loss_model, loss_grad, loss_items, unweighted_norms, weighted_norms)``
        when GradNorm is active.
    batch_interval : int
        Interval (in batches) at which running losses are printed.
    epoch_number : int
        Index of the epoch currently processed; used for progress labels
        and for computing the global training iteration.
    warmup_epoch : bool
        If ``True``, the very first epoch runs with uniform weighting to
        collect initial task losses before GradNorm takes over.
    exponential_transformation : bool
        Whether the learnable GradNorm parameters represent log-weights
        that are exponentiated and normalized before use.
    writer : torch.utils.tensorboard.SummaryWriter
        TensorBoard writer used for GradNorm diagnostics and loss curves.
    writer_bool : bool
        Enables (``True``) or disables (``False``) any TensorBoard
        logging performed inside the loop.

    Returns
    -------
    model : SuTraN
        Updated model after the epoch.
    model_optim : torch.optim.Optimizer
        Optimizer whose state reflects the latest parameter step.
    grad_optim : torch.optim.Optimizer
        Optimizer whose state reflects the latest GradNorm update.
    epoch_averages : tuple of float
        Epoch-aggregated losses. Always contains the multi-task loss,
        activity loss, and TTNE loss; remaining-runtime and/or outcome
        averages are included when those heads are present, followed by
        the final composite loss value.
    """

    # Tracking global loss over all prediction heads:
    running_loss_glb = []
    # Tracking loss of each prediction head separately: 
    running_loss_act = [] # Cross-Entropy
    running_loss_ttne = [] # MAE

    # Creating convenience flags for which heads are active 
    only_rrt = (not outcome_bool) & remaining_runtime_head
    only_out = outcome_bool & (not remaining_runtime_head)
    both_not = (not outcome_bool) & (not remaining_runtime_head)
    both = outcome_bool & remaining_runtime_head

    # binary out bool
    bin_outbool = (out_type=='binary_outcome')
    # multiclass out bool
    multic_outbool = (out_type=='multiclass_outcome')
    
    # If the only additional prediction head (on top of activity and ttne 
    # suffix) is remaining time prediction
    if only_rrt:
        running_loss_rrt = [] # MAE
        num_target_tens = 3

        # Specifying corresponding prediction targets in same order as loss items etc
        component_string_list = ['ActivitySuffix', 'TimestampSuffix', 'RemainingRuntime']

    # If the only additional prediction head (on top of activity and ttne 
    # suffix) is outcome prediction (binary or multiclass)
    elif only_out:
        running_loss_out = []
        num_target_tens = 3

        # Specifying corresponding prediction targets in same order as loss items etc
        component_string_list = ['ActivitySuffix', 'TimestampSuffix', 'Outcome']

    # If the additional prediction heads (on top of activity and ttne 
    # suffix) comprise both rrt and outcome prediction
    elif both:
        running_loss_rrt = []
        running_loss_out = []
        num_target_tens = 4

        # Specifying corresponding prediction targets in same order as loss items etc
        component_string_list = ['ActivitySuffix', 'TimestampSuffix', 'RemainingRuntime', 'Outcome']

    # If there are no additional prediction heads (on top of activity and 
    # ttne suffix)
    elif both_not: 
        num_target_tens = 2

        # Specifying corresponding prediction targets in same order as loss items etc
        component_string_list = ['ActivitySuffix', 'TimestampSuffix']

    # initializing two auxiliary counters accounting for skipped non-valid batches 
    # (possible exception handling invalid outcome batch)
    num_batches_processed = 0 
    num_batches_skipped = 0 



    for batch_num, data in tqdm(enumerate(training_loader), desc="Batch calculation at epoch {}.".format(epoch_number)):

        # Compute training iteration in terms of batch number processed (over the epochs)
        training_iteration = epoch_number * len(training_loader) + batch_num


        # out_mask can only be True if outcome_bool=True
        if out_mask: 
            inputs = data[:-(num_target_tens+1)]
            labels = data[-(num_target_tens+1):-1]
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
            inputs = data[:-num_target_tens]
            labels = data[-num_target_tens:]
            instance_mask_out = None 
        
        num_batches_processed += 1 
        # Sending inputs and labels to GPU
        inputs = [input_tensor.to(device) for input_tensor in inputs]
        labels = [label_tensor.to(device) for label_tensor in labels]

        # Restoring gradients to 0 for every batch
        model_optim.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # In case the first epoch is being proceccsed and a warmup_epoch 
        # is utilized 
        if (epoch_number==0) and warmup_epoch: 
            # Leveraging warmup epoch in which we train with equal 
            # weighting and set the initial task-specifc losses equal to 
            # the average task-specific batch losses over the batches in 
            # this first warmup epoch

            # Compute the loss 
            loss_results = loss_fn(outputs, 
                                   labels, 
                                   initial_step=True, 
                                   instance_mask_out=instance_mask_out)

            # Under these conditions, no loss_grad is returned 
            loss_model = loss_results[0]

            # Retrieving task-specific losses to track them. 
            loss_items = loss_results[1] # tuple of num_tasks individual losses (detached)



            # Compute gradients of model loss solely
            loss_model.backward()

            # Perform gradient update model parameters 
            model_optim.step()


        # In case no warmup_epoch is leveraged, but we are processing 
        # the first batch of the very first epoch
        elif (epoch_number==0) and (batch_num==0):
            
            # Restoring gradnorm gradients wrt task weights to 0. 
            grad_optim.zero_grad()

            # Not making use of warmup epoch and fixing initial 
            # task-specific losses at loss of first batch
            loss_results = loss_fn(outputs, 
                                   labels, 
                                   initial_step=True, 
                                   instance_mask_out=instance_mask_out)
            
            # Retrieving both gradient tracked losses for updating 
            # model parameters and task_weights respectively 
            loss_model = loss_results[0]
            loss_grad = loss_results[1]

            # Retrieving individual task-specific (unweighted) losses 
            # (detached) floats, for tracking
            loss_items = loss_results[2] # tuple of num_tasks individual losses (detached)


            # Retrieving numpy array containing the L2 norms of the unweighted task-specific losses 
            unweighted_gradientnorms_array = loss_results[3] # array of shape (num_tasks,)

            # Retrieving numpy array containing the L2 norms of the weighted task-specific losses 
            weighted_gradientnorms_array = loss_results[4] # array of shape (num_tasks,)


            # Compute gradients 
            total_loss = loss_model + loss_grad
            total_loss.backward()

            # Perform gradient update model parameters 
            model_optim.step()

            # Perform gradient update gradnorm task weights 
            grad_optim.step()
            
        # All other cases pertaining to default GradNorm logic.  I.e. in 
        # case we are either utilizing a warmup epoch but the first epoch 
        # has already beeen processed, or not utilizing a warmup epoch 
        # and the very first batch in the first epoch has already been 
        # processed. 
        else:
            # Restoring gradnorm gradients wrt task weights to 0. 
            grad_optim.zero_grad()

            # Compute the loss
            loss_results = loss_fn(outputs, 
                                   labels, 
                                   initial_step=False, 
                                   instance_mask_out=instance_mask_out)

            # Retrieving both gradient tracked losses for updating 
            # model parameters and task_weights respectively 
            loss_model = loss_results[0]
            loss_grad = loss_results[1]

            # Retrieving individual task-specific (unweighted) losses 
            # (detached) floats, for tracking
            loss_items = loss_results[2] # tuple of num_tasks individual losses (detached)


            # Retrieving numpy array containing the L2 norms of the unweighted task-specific losses 
            unweighted_gradientnorms_array = loss_results[3] # array of shape (num_tasks,)

            # Retrieving numpy array containing the L2 norms of the weighted task-specific losses 
            weighted_gradientnorms_array = loss_results[4] # array of shape (num_tasks,)

            # Compute gradients 
            total_loss = loss_model + loss_grad
            total_loss.backward()

            # Perform gradient update model parameters 
            model_optim.step()

            # Perform gradient update gradnorm task weights 
            grad_optim.step()

        # Tracking gradnorm progress 
        if not ((epoch_number==0) and warmup_epoch):
            # Retrieve gradnorm parameters 
            gn_parameters = loss_fn.task_weights.detach().clone().cpu().numpy() # np array of shape (num_tasks,)

            # Derive task weights used for weighting multi-task loss 
            task_weights_gn = gn_parameters.copy()

            # If we are learning the log parameters, apply exp transformation first 
            if exponential_transformation:

                # Applying softmax normalization to convert the learned 
                # log task weights to the actual normalized weights used for 
                # weighting the losses 
                task_weights_gn = np.exp(task_weights_gn - np.max(task_weights_gn)) / np.exp(task_weights_gn - np.max(task_weights_gn)).sum()
                task_weights_gn = num_target_tens * task_weights_gn # np array of shape (num_tasks,)
            
            
            else:
                task_weights_gn = num_target_tens * task_weights_gn / task_weights_gn.sum() # array of shape (num_tasks,)

            # Tracking GradNorm parameters as-is with Tensorboard 
            if writer_bool: 
                log_variables(writer=writer, 
                            type_string='GradNormParameters', 
                            component_string_list=component_string_list, 
                            scalar_value_list=gn_parameters, 
                            training_iteration=training_iteration)
                
                # Tracking (Normalized) GradNorm task weights with Tensorboard 
                log_variables(writer=writer, 
                            type_string='TaskWeightsNormed', 
                            component_string_list=component_string_list, 
                            scalar_value_list=task_weights_gn, 
                            training_iteration=training_iteration)
                
                # Tracking UNweighted norms of the gradients of the task-specific losses with 
                # regard to parameters last shared layer. 
                log_variables(writer=writer, 
                            type_string='UnweightedGradientNorms', 
                            component_string_list=component_string_list, 
                            scalar_value_list=unweighted_gradientnorms_array, 
                            training_iteration=training_iteration)
                
                # Tracking WEIGHTED norms of the gradients of the task-specific losses with 
                # regard to parameters last shared layer. 
                log_variables(writer=writer, 
                            type_string='WeightedGradientNorms', 
                            component_string_list=component_string_list, 
                            scalar_value_list=weighted_gradientnorms_array, 
                            training_iteration=training_iteration)
                
                # Tracking gradnorm loss 
                writer.add_scalar('Loss/GradNorm', loss_grad.item(), training_iteration)



        # Tracking composite loss and its task-specific components 
        if writer_bool: 
            #   Composite loss 
            writer.add_scalar('Loss/Total', loss_model.item(), training_iteration)

            log_variables(writer=writer, 
                            type_string='Loss', 
                            component_string_list=component_string_list, 
                            scalar_value_list=loss_items, 
                            training_iteration=training_iteration)

        # Tracking losses and metrics
        running_loss_glb.append(loss_model.item())

        running_loss_act.append(loss_items[0])
        # Log loss evolution activity suffix 
        
        # Log loss evolution timestamp suffix prediction
        running_loss_ttne.append(loss_items[1])

        if only_rrt:
            # Log loss evolution remaining runtime prediction
            running_loss_rrt.append(loss_items[-1])

        elif only_out: 
            # Log loss evolution outcome prediction (binary or multiclass)
            running_loss_out.append(loss_items[-1])

        
        elif both:
            # Log loss evolution Remaining Runtime prediction
            running_loss_rrt.append(loss_items[-2])

            # Log loss evolution outcome prediction (binary or multiclass)
            running_loss_out.append(loss_items[-1])






        
        if batch_num % batch_interval == (batch_interval-1):
                print("------------------------------------------------------------")
                print("Epoch {}, batch {}:".format(epoch_number, batch_num))
                print("Running average global loss: {} (over last {} batches)".format(sum(running_loss_glb[-batch_interval:])/batch_interval, batch_interval))
                print("Running average activity prediction loss: {} (Cross Entropy over last {} batches)".format(sum(running_loss_act[-batch_interval:])/batch_interval, batch_interval))
                print("Running average time till next event prediction loss: {} (MAE over last {} batches)".format(sum(running_loss_ttne[-batch_interval:])/batch_interval, batch_interval))
                if remaining_runtime_head:
                    print("Running average (complete) remaining runtime prediction loss: {} (MAE over last {} batches)".format(sum(running_loss_rrt[-batch_interval:])/batch_interval, batch_interval))
                if outcome_bool:
                    if bin_outbool:
                        print("Running average binary outcome prediction loss: {} (BCE over last {} batches)".format(sum(running_loss_out[-batch_interval:])/batch_interval, batch_interval))

                    elif multic_outbool:
                        print("Running average MC outcome prediction loss: {} (CE over last {} batches)".format(sum(running_loss_out[-batch_interval:])/batch_interval, batch_interval))
                print("------------------------------------------------------------")

    print("=======================================")
    print("End of epoch {}".format(epoch_number))
    print("=======================================")
    num_batches = len(running_loss_glb)

    # Global loss 
    last_running_avg_glob = sum(running_loss_glb[-batch_interval:])/batch_interval
    average_global_loss_epoch = sum(running_loss_glb) / num_batches
    print("Running average global loss: {} (over last {} batches)".format(last_running_avg_glob, batch_interval))
    print("Average Multi-Task loss over all batches this epoch: {}".format(average_global_loss_epoch))

    # Activity suffix prediction loss (cat. CE)
    last_running_avg_act = sum(running_loss_act[-batch_interval:])/batch_interval
    average_global_loss_act = sum(running_loss_act) / num_batches
    print("Running average activity prediction loss: {} (Cross Entropy over last {} batches)".format(last_running_avg_act, batch_interval))
    print("Average activity suffix loss over all batches this epoch: {}".format(average_global_loss_act))


    # Timestamp suffix prediction loss (MAE)
    last_running_avg_ttne = sum(running_loss_ttne[-batch_interval:])/batch_interval
    average_global_loss_ttne = sum(running_loss_ttne) / num_batches
    print("Running average time till next event prediction loss: {} (MAE over last {} batches)".format(last_running_avg_ttne, batch_interval))
    print("Average timestamp suffix loss over all batches this epoch: {}".format(average_global_loss_ttne))

    # Remaining Runtime prediction loss (MAE)  
    if only_rrt:
        last_running_avg_rrt = sum(running_loss_rrt[-batch_interval:])/batch_interval
        average_global_loss_rrt = sum(running_loss_rrt) / num_batches
        print("Running average (complete) remaining runtime prediction loss: {} (MAE over last {} batches)".format(last_running_avg_rrt, batch_interval))
        print("Average remaining runtime loss over all batches this epoch: {}".format(average_global_loss_rrt))

        epoch_averages = average_global_loss_epoch, average_global_loss_act, average_global_loss_ttne, average_global_loss_rrt, loss_model
    
    # Outcome prediction loss 
    elif only_out:
        last_running_avg_out = sum(running_loss_out[-batch_interval:])/batch_interval
        average_global_loss_out = sum(running_loss_out) / num_batches
        if bin_outbool:
            print("Running average binary outcome prediction loss: {} (BCE over last {} batches)".format(last_running_avg_out, batch_interval))
        elif multic_outbool: 
            print("Running average MC outcome prediction loss: {} (CE over last {} batches)".format(last_running_avg_out, batch_interval))
        epoch_averages = average_global_loss_epoch, average_global_loss_act, average_global_loss_ttne, average_global_loss_out, loss_model
    
    # Remaining Runtime prediction loss (MAE) and Outcome prediction loss 
    elif both:
        # Remaining runtime 
        last_running_avg_rrt = sum(running_loss_rrt[-batch_interval:])/batch_interval
        average_global_loss_rrt = sum(running_loss_rrt) / num_batches
        print("Running average (complete) remaining runtime prediction loss: {} (MAE over last {} batches)".format(last_running_avg_rrt, batch_interval))
        print("Average remaining runtime loss over all batches this epoch: {}".format(average_global_loss_rrt))

        # outcome 
        last_running_avg_out = sum(running_loss_out[-batch_interval:])/batch_interval
        average_global_loss_out = sum(running_loss_out) / num_batches
        if bin_outbool:
            print("Running average binary outcome prediction loss: {} (BCE over last {} batches)".format(last_running_avg_out, batch_interval))
            print("Average binary outcome prediction loss over all batches this epoch: {}".format(average_global_loss_out))
        elif multic_outbool: 
            print("Running average MC outcome prediction loss: {} (CE over last {} batches)".format(last_running_avg_out, batch_interval))
            print("Average MC outcome prediction loss over all batches this epoch: {}".format(average_global_loss_out))

        epoch_averages = average_global_loss_epoch, average_global_loss_act, average_global_loss_ttne, average_global_loss_rrt, average_global_loss_out, loss_model
    
    elif both_not:
        epoch_averages = average_global_loss_epoch, average_global_loss_act, average_global_loss_ttne, loss_model


    # In case of leveraging a warmup_epoch and just having finished this first epoch, 
    # set the initial task-specific losses to the averages over the first epoch 
    if (epoch_number==0) and warmup_epoch: 
        # Subsetting list of individual task losses 
        average_task_losses = epoch_averages[1:-1]

        # Creating tensor out of it 
        init_losses_tensor = torch.tensor(data=average_task_losses, dtype=torch.float32, device=device) # (num_tasks, )
        
        # Setting initial losses 
        loss_fn.set_initial_losses(init_losses_tensor)


    if out_mask: 
        print("Number of batches skipped due to no valid outcome instances: {}".format(num_batches_skipped))

    return model, model_optim, grad_optim, epoch_averages         




def train_model(model, 
                train_dataset, 
                val_dataset, 
                start_epoch, 
                num_epochs, 
                remaining_runtime_head, 
                outcome_bool, 
                num_classes, 
                batch_interval, 
                path_name, 
                num_categoricals_pref, 
                mean_std_ttne, 
                mean_std_tsp, 
                mean_std_tss, 
                mean_std_rrt, 
                batch_size, 
                clen_dis_ref, 
                og_caseint_train, 
                og_caseint_val,
                median_caselen,
                shared_params, 
                out_mask=False, 
                instance_mask_out_train=None, 
                instance_mask_out_val=None, 
                out_type=None, 
                num_outclasses=None,
                lr_model=0.0002, 
                lr_gradnorm=0.025, # default implementation Chen et al. 
                alpha=1.5, 
                exponential_transformation=False, 
                warmup_epoch=False, 
                theoretical_initial_loss=False,
                patience=24, 
                best_MAE_ttne=1e9, 
                best_DL_sim=-1, 
                best_MAE_rrt=1e9, 
                best_BCE=1e9, 
                best_auc_pr=-1,
                best_auc_roc=-1,
                best_CE_MCO=1e9, 
                best_macro_F1=-1, 
                best_weighted_F1=-1, 
                writer_bool=False,
                seed=None
                ):
    """Outer training loop SuTraN, leveraging the Multi-Task Optimization 
    (MTO) technique "GradNorm" by Chen et al. [1]_. 

    Parameters
    ----------
    model : SuTraN
        SuTraN model currently in training mode. Forward calls must return
        the tuple of task outputs expected by ``loss_fn``.
    train_dataset : tuple of torch.Tensor
        Tuple containing the tensors comprising the training set. This 
        includes, i.a., the labels. All tensors have an outermost 
        dimension of the same size, i.e. `N_train`, the number of 
        original training set instances / prefix-suffix pairs. 
    val_dataset : tuple of torch.Tensor 
        Tuple containing the tensors comprising the validation set. This 
        includes, i.a., the labels. All tensors have an outermost 
        dimension of the same size, i.e. `N_val`, the number of 
        original validation set instances / prefix-suffix pairs. 
    start_epoch : int
        Number of the epoch from which the training loop is started. 
        First call to ``train_model()`` should be done with 
        ``start_epoch=0``.
    num_epochs : int
        Number of epochs to train. When resuming training with another 
        loop of num_epochs, for the new ``train_model()``, the new 
        ``start_epoch`` argument should be equal to the current one 
        plus the current value for ``num_epochs``.
    remaining_runtime_head : bool
        Whether or not the model is also trained to directly predict the 
        remaining runtime given a prefix. 
    outcome_bool : bool, optional 
        Whether an outcome prediction head is trained (binary or
        multiclass, depending on ``out_type``). If 
        `outcome_bool=True`, a prediction head for predicting 
        the outcome given a prefix is added. This prediction 
        head, in contrast to the time till next event and activity 
        suffix predictions, will only be trained to provide a 
        prediction at the first decoding step. Note that the 
        value of `outcome_bool` should be aligned with the 
        `outcome_bool` parameter of the model and train 
        procedure, as well as with the preprocessing pipeline that 
        produces the labels. By default `False`.
    num_classes : int
        The number of output neurons for the activity prediction head. 
        This includes the padding token (0) and the END token. 
    batch_interval : int
        The periodic amount of batches trained for which the moving average 
        losses and metrics are printed and recorded. E.g. if 
        ``batch_interval=100``, then after every 100 batches, the 
        moving averages of all metrics and losses during training are 
        recorded, printed and reset to 0. 
    path_name : str 
        Needed for saving results and callbacks in the 
        appropriate subfolders. This is the path name 
        of the subfolder for which all the results 
        and callbacks (model copies) should be 
        stored for the current event log and 
        model configuration.
    num_categoricals_pref : int
        The number of categorical features (including the activity label) 
        contained within each prefix event token.
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
        Note, only required in case `remaining_runtime_head=True`. 
        Mean is the first entry, std the 2nd.
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
    shared_params : model.shared_layer.parameters()
        Parameters of the final shared layer whose gradients feed 
        GradNorm's task-norm computation. Pass the actual parameter 
        iterator (no copies) so optimizer updates remain in sync.
    out_mask : bool, optional
        If `True`, each batch includes an outcome mask tensor to suppress 
        instances whose prefixes already reveal the label from 
        contributing to the outcome loss/metrics. Defaults to `False`.
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
        is required (`outcome_bool=False`), or in case outcome 
        prediction is required, but the outcome labels are not derived 
        directly from information contained in other event data, and 
        hence not contained in part of the `N_train` training instances' 
        prefixes (model inputs) (`outcome_bool=True` and 
        `out_mask=False`), the default of `None` should be retained. 
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
        is required (`outcome_bool=False`), or in case outcome 
        prediction is required, but the outcome labels are not derived 
        directly from information contained in other event data, and 
        hence not contained in part of the `N_val` validation instances' 
        prefixes (model inputs) (`outcome_bool=True` and 
        `out_mask=False`), the default of `None` should be retained. 
    out_type : {None, 'binary_outcome', 'multiclass_outcome'}, optional
        The type of outcome prediction that is being performed in the 
        multi-task setting. Only taken into account of outcome prediction 
        is included in the event log to begin with, and hence if 
        `outcome_bool=True`. If so, `'binary_outcome'` denotes binary 
        outcome (BO) prediction (binary classification), while 
        `'multiclass_outcome'` denotes multi-class outcome (MCO) 
        prediction (Multi-Class classification). 
    num_outclasses : {int, None}, optional
        The number of outcome classes in case 
        `outcome_bool=True` and `out_type='multiclass_outcome'`. By 
        default `None`. 
    lr_model : float, optional 
        Initial learning rate for training the model parameters. By 
        default 0.0002. 
    lr_gradnorm : float, optional 
        Learning rate for training the gradnorm parameters (i.e. the task 
        weights). By default 0.025.
    alpha : float 
        Strength/restoring-force parameter for GradNorm (default 1.5).
    exponential_transformation : bool 
        Boolean parameter pertaining to GradNorm optimization. 
        If `True`, we will learn and update the natural logarithm of 
        the weights instead of directly learning the task weights. 
        This corresponds to applying softmax normalization 
        (w = T * softmax(w)) instead of default normalization 
        (w_i = T * w_i / w.sum()) and hence initializing the 
        learnable gradnorm parameters as a 0-vector of shape 
        num_tasks, instead of a vector filled with ones. 
    warmup_epoch : bool, optional
        Boolean parameter pertaining to GradNorm optimization. 
        If `True`, a warmup epoch is utilized in which we 
        just apply equal weighting and only update the model 
        parameters based on the equally weighted composite loss. 
        The initial losses will be set to the average of the 
        individual losses over all batches processed during this 
        first warmup epoch. Utilizing a warmup epoch is optional, and 
        would deviate from the original GradNorm implementation, with 
        the aim of obtaining more robust initial losses. 
        By default `False`.
    theoretical_initial_loss : bool, optional
        Boolean parameter pertaining to GradNorm optimization. 
        If `True`, we are going to use a theoretical initial loss for 
        the categorical cross entropy loss for activity suffix 
        prediction of log(num_classes), and in case outcome 
        prediction is performed as well, also a theoretical initial 
        loss for binary or multi-class crossentropy as well. 
        By default `False`.
    patience : int, optional. 
        Max number of epochs without any improvement in any of the 
        validation metrics. After `patience` epochs without any 
        improvement, the training loop is terminated early. By default 24.
    best_MAE_ttne : float, optional
        Best validation Mean Absolute Error for the time till 
        next event suffix prediction. The defaults apply if the training 
        loop is initialized for the first time for a given configuration. 
        If the training loop is resumed from a certain checkpoint, the 
        best results of the previous training loop should be given. 
    best_DL_sim : float, optional
        Best validation 1-'normalized Damerau-Levenshtein distance for 
        activity suffix prediction so far. The defaults apply if the  
        training loop is initialized for the first time for a given 
        configuration. If the training loop is resumed from a certain 
        checkpoint, the best results of the previous training loop should 
        be given. 
    best_MAE_rrt : float, optional
        Best validation Mean Absolute Error for the remaining runtime 
        prediction so far. The defaults apply if the training 
        loop is initialized for the first time for a given configuration. 
        If the training loop is resumed from a certain checkpoint, the 
        best results of the previous training loop should be given. If 
        `remaining_runtime_head=False`, the defaults can be retained even 
        when resuming the training loop from a checkpoint. 
    best_BCE : float, optional
        Best validation Binary Cross Entropy binary outcome prediction so 
        far. The defaults apply if the training loop is initialized for 
        the first time for a given configuration. If the training loop is 
        resumed from a certain checkpoint, the best results of the 
        previous training loop should be given. If 
        `outcome_bool=False`, the defaults can be retained even when 
        resuming the training loop from a checkpoint. 
    best_auc_pr : float, optional
        Best validation score for area under the Precision Recall curve  
        so far (outcome prediction). The defaults apply if the training 
        loop is initialized for the first time for a given configuration.  
        If the training loop is resumed from a certain checkpoint, the 
        best results of the previous training loop should be given. If 
        `outcome_bool=False`, the defaults can be retained even when 
        resuming the training loop from a checkpoint. 
    best_auc_roc : float, optional
        Best validation Area Under the ROC Curve observed so far for the 
        outcome task. Initialize to the default when starting from 
        scratch; provide the stored value when resuming training.
    best_CE_MCO : float, optional
        Best validation cross-entropy for multiclass outcome prediction 
        so far. Defaults apply on a fresh run; supply the prior best when 
        resuming.
    best_macro_F1 : float, optional
        Best validation macro-averaged F1 score for multiclass outcome 
        prediction. Use the default for new runs and the saved score when 
        resuming.
    best_weighted_F1 : float, optional
        Best validation class-weighted F1 score for multiclass outcome 
        prediction. Defaults on fresh runs; restore the previous best 
        when continuing training.
    writer_bool : bool, optional
        Enables TensorBoard logging of GradNorm diagnostics and loss 
        curves when True. Logging adds overhead, so the default is False 
        for lightweight experimentation.
    seed : {int, None}, optional
        Random seed used to reproduce epoch-level shuffling, GradNorm 
        initialization, and warm-up behavior. If None, the loop derives 
        seeds from the epoch index.


    Returns
    -------
    None
        Saves checkpoints/CSV traces; no explicit return value. The trained 
        model can be retrieved from the saved checkpoints. 

    References
    ----------
    .. [1] Chen, Z., Badrinarayanan, V., Lee, C. Y., & Rabinovich, A. (2018).
           GradNorm: Gradient normalization for adaptive loss balancing in deep
           multitask networks. In ICML (pp. 794-803). PMLR.
    """
    if writer_bool:
        writer = SummaryWriter(log_dir=path_name)
    else: 
        writer = None

    # Checking whether GPU is being used
    print("Device: {}".format(device))

    # Assigning model to GPU. 
    model.to(device)

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
        
        if out_mask:
            if instance_mask_out_train is None or not isinstance(instance_mask_out_train, torch.Tensor):
                raise ValueError(
                    "When `outcome_bool=True` and `out_mask=True`, "
                    "'instance_mask_out_train' must be a torch.Tensor and not None."
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
    train_losses_global = []
    train_losses_act = []
    train_losses_ttne = []

    # Track evolution of validation metrics over the epoch loop by initializing empty lists. 
    avg_MAE_ttne_stand_glob, avg_MAE_ttne_minutes_glob, avg_dam_lev_glob = ([] for _ in range(3))


    if remaining_runtime_head:
        avg_MAE_stand_RRT_glob, avg_MAE_minutes_RRT_glob = [], []
    if outcome_bool: 
        if bin_outbool:
            avg_BCE_out_glob, avg_auc_roc_glob, avg_auc_pr_glob = [], [], []
            acc_glob, f1_glob, precision_glob, recall_glob, balanced_accuracy_glob = [], [], [], [], []
        elif multic_outbool: 
            avg_CE_MCO_glob, acc_glob, macro_f1_glob, weighted_f1_glob = [], [], [], []
            macro_precision_glob, weighted_precision_glob, macro_recall_glob, weighted_recall_glob = [], [], [], []

    # Creating convenience flags for which heads are active
    only_rrt = (not outcome_bool) & remaining_runtime_head
    only_out = outcome_bool & (not remaining_runtime_head)
    both_not = (not outcome_bool) & (not remaining_runtime_head)
    both = outcome_bool & remaining_runtime_head

    # Initialize lists for keeping track of training losses of optional 
    # prediction heads if included. 
    if remaining_runtime_head:
        train_losses_rrt = []
    if outcome_bool:
        train_losses_out = []

    # Specifying composite loss function  
    loss_fn = MultiOutputLoss_GradNorm(num_classes, 
                                       remaining_runtime_head, 
                                       outcome_bool, 
                                       clen_dis_ref, 
                                       shared_params, 
                                       alpha, 
                                       exponential_transformation, 
                                       warmup_epoch, 
                                       theoretical_initial_loss, 
                                       out_mask, 
                                       out_type=out_type, 
                                       num_outclasses=num_outclasses)

    # Specifying optimizers and learning rate scheduler
    decay_factor = 0.96
    model_optim = torch.optim.AdamW(model.parameters(), lr=lr_model, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=model_optim, gamma=decay_factor)

    grad_optim = torch.optim.Adam([loss_fn.task_weights],lr = lr_gradnorm) 




    num_epochs_not_improved = 0
    for epoch in range(start_epoch, start_epoch + num_epochs):

        if warmup_epoch and (epoch==1):
            # Temporary setting the manual_seed to original seed value 
            # to ensure that the optimizer and scheduler is initialized 
            # with the same seed as other techniques for fair comparison.
            if seed is not None:
                torch.manual_seed(seed)
            else:
                torch.manual_seed(24)
            # Reinitializing the optimizer for the model weights since it will now 
            # transition to a different training setup (gradnorm)
            model_optim = torch.optim.AdamW(model.parameters(), lr=lr_model, weight_decay=0.0001)
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=model_optim, gamma=decay_factor)

        print(" ")
        print("------------------------------------")
        print('EPOCH {}:'.format(epoch))
        print("____________________________________")

        if seed is not None:
            # Setting seed for reproducible shuffling each epoch
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

            # Setting seed for reproducible shuffling each epoch
            torch.manual_seed(seed_value) 
            train_dataloader = DataLoader(train_tens_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        else: 
            # Setting seed for reproducible shuffling each epoch
            torch.manual_seed(seed_value) 
            train_dataloader = DataLoader(train_tens_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        # Activate gradient tracking
        model.train(True)

        model, model_optim, grad_optim, epoch_averages = train_epoch(model, 
                                                                     train_dataloader, 
                                                                     remaining_runtime_head,
                                                                     outcome_bool,  
                                                                     out_mask,
                                                                     out_type,
                                                                     model_optim, 
                                                                     grad_optim, 
                                                                     loss_fn, 
                                                                     batch_interval, 
                                                                     epoch, 
                                                                     warmup_epoch,
                                                                     exponential_transformation, 
                                                                     writer, 
                                                                     writer_bool)
        train_losses_global.append(epoch_averages[0])
        train_losses_act.append(epoch_averages[1])
        train_losses_ttne.append(epoch_averages[2])
        last_loss = epoch_averages[-1]
        if only_rrt:
            train_losses_rrt.append(epoch_averages[3])
        elif only_out:
            train_losses_out.append(epoch_averages[3])
        elif both:
            train_losses_rrt.append(epoch_averages[3])
            train_losses_out.append(epoch_averages[4])

        # Set the model to evaluation mode and disabling dropout before 
        # validation (inference)
        model.eval()

        if clen_dis_ref:
            # Second tuple element contains list of CB metrics
            _, inf_results, _ = inference_loop(model=model, 
                                               inference_dataset=val_dataset,
                                               remaining_runtime_head=remaining_runtime_head, 
                                               outcome_bool=outcome_bool, 
                                               out_mask=out_mask,
                                               out_type=out_type, 
                                               num_outclasses=num_outclasses, 
                                               num_categoricals_pref=num_categoricals_pref, 
                                               mean_std_ttne=mean_std_ttne, 
                                               mean_std_tsp=mean_std_tsp, 
                                               mean_std_tss=mean_std_tss, 
                                               mean_std_rrt=mean_std_rrt, 
                                               og_caseint=og_caseint_val,
                                               instance_mask_out=instance_mask_out_val,
                                               results_path=None, 
                                               val_batch_size=4096)
        else: 
            # First tuple element contains list of IB (default) metrics
            inf_results, _, _ = inference_loop(model=model, 
                                               inference_dataset=val_dataset,
                                               remaining_runtime_head=remaining_runtime_head, 
                                               outcome_bool=outcome_bool, 
                                               out_mask=out_mask,
                                               out_type=out_type, 
                                               num_outclasses=num_outclasses, 
                                               num_categoricals_pref=num_categoricals_pref, 
                                               mean_std_ttne=mean_std_ttne, 
                                               mean_std_tsp=mean_std_tsp, 
                                               mean_std_tss=mean_std_tss, 
                                               mean_std_rrt=mean_std_rrt, 
                                               og_caseint=og_caseint_val,
                                               instance_mask_out=instance_mask_out_val,
                                               results_path=None, 
                                               val_batch_size=4096)

        # TTNE MAE metrics
        avg_MAE_ttne_stand, avg_MAE_ttne_minutes = inf_results[:2]
        # Average Normalized Damerau-Levenshtein similarity Activity Suffix 
        # prediction
        avg_dam_lev = inf_results[2]

        avg_MAE_ttne_stand_glob.append(avg_MAE_ttne_stand)
        avg_MAE_ttne_minutes_glob.append(avg_MAE_ttne_minutes)
        avg_dam_lev_glob.append(avg_dam_lev)

        print("Avg MAE TTNE prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_ttne_stand, avg_MAE_ttne_minutes))
        print("Avg 1-(normalized) DL distance acitivty suffix prediction validation set: {}".format(avg_dam_lev))

        # Determining whether one of the primary validation 
        # metrics has been improved compared to the current best value. 
        better = False
        if avg_MAE_ttne_stand < best_MAE_ttne: 
            better = True
            best_MAE_ttne = avg_MAE_ttne_stand
        if avg_dam_lev > best_DL_sim:
            better = True 
            best_DL_sim = avg_dam_lev

        
        if only_rrt:
            # MAE standardized RRT predictions 
            avg_MAE_stand_RRT = inf_results[3]
            # MAE RRT converted to minutes
            avg_MAE_minutes_RRT = inf_results[4]

        elif only_out:
            if bin_outbool: # binary outcome prediction metrics
                # Binary Cross Entropy outcome prediction
                avg_BCE_out = inf_results[3]
                # AUC-ROC outcome prediction
                auc_roc = inf_results[4]
                # AUC-PR outcome prediction
                auc_pr = inf_results[5]
                
                binary_dict = inf_results[6]
            
            elif multic_outbool: 
                # Categorical Cross Entropy Inf set MCO prediction
                avg_CE_MCO = inf_results[3]
                # Dict of additional validation metrics 
                mc_dict = inf_results[4]

        elif both: 
            # MAE standardized RRT predictions 
            avg_MAE_stand_RRT = inf_results[3]
            # MAE RRT converted to minutes
            avg_MAE_minutes_RRT = inf_results[4]

            if bin_outbool: # binary outcome prediction metrics
                # Binary Cross Entropy outcome prediction
                avg_BCE_out = inf_results[5]
                # AUC-ROC outcome prediction
                auc_roc = inf_results[6]
                # AUC-PR outcome prediction
                auc_pr = inf_results[7]
                
                binary_dict = inf_results[8]

            
            elif multic_outbool: # Multi-Class Outcome (MCO) prediction metrics 
                # Categorical Cross Entropy Inf set MCO prediction
                avg_CE_MCO = inf_results[5]
                # Dict of additional validation metrics 
                mc_dict = inf_results[6]


        if remaining_runtime_head:
            if avg_MAE_stand_RRT < best_MAE_rrt: 
                better = True
                best_MAE_rrt = avg_MAE_stand_RRT
            print("Avg MAE RRT prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_stand_RRT, avg_MAE_minutes_RRT))
            # Store evolution validation measures RRT: 
            avg_MAE_stand_RRT_glob.append(avg_MAE_stand_RRT)
            avg_MAE_minutes_RRT_glob.append(avg_MAE_minutes_RRT)

        if bin_outbool: 
            avg_BCE_out_glob.append(avg_BCE_out)

            avg_auc_roc_glob.append(auc_roc)

            avg_auc_pr_glob.append(auc_pr)

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
        
        if multic_outbool: 
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

        if better == False: 
            num_epochs_not_improved += 1
        else:
            num_epochs_not_improved = 0

        # Saving checkpoint every epoch

        model_path = os.path.join(path_name, 'model_epoch_{}.pt'.format(epoch))
        checkpoint = {'epoch:' : epoch, 
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': model_optim.state_dict(), 
                        'grad_optim_state_dict': grad_optim.state_dict(), 
                        'loss': last_loss}
        torch.save(checkpoint, model_path)
            
        # Update the learning rate
        lr_scheduler.step()
            
        torch.cuda.empty_cache()


        if num_epochs_not_improved >= patience:
            print("No improvements in validation loss for {} consecutive epochs. Final epoch: {}".format(patience, epoch))
            break

        # Logging validation metrics with tensorboard 
        if writer_bool:
            #   DL distance 
            writer.add_scalar('ValidationMetrics/DLS', avg_dam_lev, epoch)

            #   MAE TTNE (in minutes)
            writer.add_scalar('ValidationMetrics/MAE_TTNE_MIN', avg_MAE_ttne_minutes, epoch)

            #   MAE RRT (in minutes)
            if remaining_runtime_head:
                writer.add_scalar('ValidationMetrics/MAE_RRT_MIN', avg_MAE_minutes_RRT, epoch)
            
            #   Outcome metrics 
            if outcome_bool:
                if bin_outbool:
                    writer.add_scalar('ValidationMetrics/auc_roc', auc_roc, epoch)
                    writer.add_scalar('ValidationMetrics/auc_pr', auc_pr, epoch)
                    writer.add_scalar('ValidationMetrics/avg_BCE', avg_BCE_out, epoch)
                
                elif multic_outbool:
                    writer.add_scalar('ValidationMetrics/avg_CE_MCO', avg_CE_MCO, epoch)
                    writer.add_scalar('ValidationMetrics/macro_f1', macro_f1, epoch)
                    writer.add_scalar('ValidationMetrics/weighted_f1', weighted_f1, epoch)
        



    # Writing training progress to csv at the end of the current training loop
    # Persist per-epoch losses/metrics for post-run analysis and checkpoint audits.
    # Extra columns are appended on demand (RRT and/or outcome statistics) to mirror
    # whichever heads were active in this training run.

    results_path = os.path.join(path_name, 'backup_results.csv')
    epoch_list = [i for i in range(len(train_losses_global))]

    results = pd.DataFrame(data = {'epoch' : epoch_list, 
                        'composite training loss' : train_losses_global, 
                        'activity training loss (cross entropy)': train_losses_act, 
                        'time till next event training loss (MAE)': train_losses_ttne, 
                        'TTNE - standardized MAE validation': avg_MAE_ttne_stand_glob, 
                        'TTNE - minutes MAE validation': avg_MAE_ttne_minutes_glob, 
                        'Activity suffix: 1-DL (validation)': avg_dam_lev_glob})
    
    # Adding additional train and val metrics to df depending on set of PPM tasks
    if remaining_runtime_head:
        results['(complete) remaining runtime training loss (MAE)'] = train_losses_rrt
        results['RRT - standardized MAE validation'] = avg_MAE_stand_RRT_glob
        results['RRT - mintues MAE validation'] = avg_MAE_minutes_RRT_glob
    
    if bin_outbool:
        results['outcome prediction training loss (BCE)'] = train_losses_out
        results['Binary Outcome - BCE validation'] = avg_BCE_out_glob
        results['Binary Outcome - AUC-ROC validation'] = avg_auc_roc_glob
        results['Binary Outcome - AUC-PR validation'] = avg_auc_pr_glob
        results['Binary Outcome - Accuracy'] = acc_glob
        results['Binary Outcome - F1 score'] = f1_glob
        results['Binary Outcome - Precision'] = precision_glob
        results['Binary Outcome - Recall'] = recall_glob
        results['Binary Outcome - Balanced Accuracy'] = balanced_accuracy_glob
    
    if multic_outbool: 
        results['Outcome Prediction (MCO) training loss (CE)'] = train_losses_out
        results['Multi-Class Outcome - CE validation'] = avg_CE_MCO_glob
        results['Multi-Class Outcome - Accuracy'] = acc_glob
        results['Multi-Class Outcome - Macro-F1 score'] = macro_f1_glob
        results['Multi-Class Outcome - Weighted-F1 score'] = weighted_f1_glob
        results['Multi-Class Outcome - Macro-Precision'] = macro_precision_glob
        results['Multi-Class Outcome - Weighted-Precision'] = weighted_precision_glob
        results['Multi-Class Outcome - Macro-Recall'] = macro_recall_glob
        results['Multi-Class Outcome - Weighted-Recall'] = weighted_recall_glob

    results.to_csv(results_path, index=False)