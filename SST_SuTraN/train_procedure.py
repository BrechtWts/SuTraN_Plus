import torch
import torch.nn as nn

# from SuTraN.train_utils import MultiOutputLoss
from SST_SuTraN.train_utils import SST_Loss
from tqdm import tqdm
import os
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from SST_SuTraN.inference_procedure import inference_loop

# Importing functionality for Uniform Case-Based Sampling (UCBS) 
# (part of CaLenDiR training)
from CaLenDiR_Utils.case_based_sampling import sample_train_instances, precompute_indices

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, 
                training_loader, 
                seq_task,
                optimizer,
                loss_fn, 
                epoch_number, 
                max_norm):
    """Run one SST training epoch over the current dataloader.

    Parameters
    ----------
    model : SuTraN_SST
        Model in training mode whose parameters will be updated.
    training_loader : torch.utils.data.DataLoader
        Loader supplying SST batches (prefix inputs plus a single
        sequential label tensor).
    seq_task : {'activity_suffix', 'timestamp_suffix'}
        Active sequential task, used both for logging and to interpret
        label shapes.
    optimizer : torch.optim.Optimizer
        Optimizer wrapping `model.parameters()`.
    loss_fn : SST_Loss
        Task-specific loss (masked CE or masked MAE).
    epoch_number : int
        Epoch index used for logging.
    max_norm : float
        Max gradient norm used for clipping.

    Returns
    -------
    tuple
        `(model, optimizer, (avg_loss, avg_grad_norm, avg_clipped_norm))`
        with epoch-average metrics for the outer training loop.
    """


    # Tracking loss of sequential prediction task at hand (activity or
    # timestamp suffix prediction)
    running_loss = []

    if seq_task == 'activity_suffix':
        loss_string = 'Cross Entropy'
    elif seq_task == 'timestamp_suffix':
        loss_string = 'MAE'

    original_norm_glb = []
    clipped_norm_glb = []


    for batch_num, data in tqdm(enumerate(training_loader), desc="Batch calculation at epoch {}.".format(epoch_number)):
        # Subsetting inputs and labels from the data tensor
        inputs = data[:-1]
        labels = data[-1] # in case of timestamp suffix prediction, 
        #                   this is the target tensor for the ttne suffix, 
        #                   shape (batch_size, window_size, 1) and dtype
        #                   torch.float32.
        #                   In case of activity suffix prediction, this is
        #                   the target tensor for the activity suffix, shape 
        #                   (batch_size, window_size) and dtype torch.int64.

        # Sending inputs and labels to GPU
        inputs = [input_tensor.to(device) for input_tensor in inputs]
        # labels = [label_tensor.to(device) for label_tensor in labels]
        labels = labels.to(device)

        # Restoring gradients to 0 for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss 
        loss = loss_fn(outputs, labels)

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
    print("Average {} loss {} prediction over all batches this epoch: {}".format(loss_string, seq_task, average_global_loss_epoch))

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
                seq_task, 
                num_classes, 
                path_name, 
                num_categoricals_pref, 
                mean_std_ttne, 
                mean_std_tsp, 
                mean_std_tss, 
                batch_size, 
                clen_dis_ref, 
                og_caseint_train, 
                og_caseint_val,
                median_caselen,
                patience = 24, 
                lr_scheduler_present=False, 
                lr_scheduler=None, 
                best_MAE_ttne=1e9, 
                best_DL_sim=-1, 
                max_norm = 2., 
                seed=None):
    """Outer training loop SuTraN_SST, i.e. the Sequential, Single-Task 
    version of SuTraN, to be trained for either activity suffix or 
    timestamp suffix prediction (`seq_task`). 

    Parameters
    ----------
    model : SuTraN_SST
        The initialized and current version of a Sequential, Single-Task
        (SST) version of SuTraN for sequential single-task PPM, 
        predicting solely the activity suffix or timestamp suffix.
    optimizer : torch optimizer
        torch.optim.AdamW optimizer. Should already be initialized and 
        wrapped around the parameters of `model`. 
    train_dataset : tuple of torch.Tensor
        Tuple of tensors already subset for the chosen SST head
        (via `SST_SuTraN.convert_tensordata.subset_data`),
        so only the relevant sequential label tensor remains alongside
        the shared prefix/suffix inputs. Every tensor has leading
        dimension `N_train`.
    val_dataset : tuple of torch.Tensor
        Validation counterpart of `train_dataset`, likewise pre-subset
        so the trailing tensor matches `seq_task`. Leading dimension is
        `N_val`.
    start_epoch : int
        Number of the epoch from which the training loop is started. 
        First call to ``train_model()`` should be done with 
        ``start_epoch=0``.
    num_epochs : int
        Number of epochs to train. When resuming training with another 
        loop of num_epochs, for the new ``train_model()``, the new 
        ``start_epoch`` argument should be equal to the current one 
        plus the current value for ``num_epochs``.
    seq_task : {'activity_suffix', 'timestamp_suffix'}
        The (sole) sequential prediction task trained and evaluated 
        for. 
    num_classes : int or None
        In case of `seq_task='activity_suffix'`, this should be the number
        of output neurons for the activity prediction head. This includes 
        the padding token (0) and the END token. In case of
        `seq_task='timestamp_suffix'`, this should be None.
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
        std the second. Ignored if `seq_task='activity_suffix'`.
    mean_std_tsp : list of float
        Training mean and standard deviation used to standardize the time 
        since previous event (in seconds) feature of the decoder suffix 
        tokens. Needed for re-converting time since previous event values 
        to original scale (seconds). Mean is the first entry, std the 2nd.
        Ignored if `seq_task='activity_suffix'`.
    mean_std_tss : list of float
        Training mean and standard deviation used to standardize the time 
        since start (in seconds) feature of the decoder suffix tokens. 
        Needed for re-converting time since start to original scale 
        (seconds). Mean is the first entry, std the 2nd. 
        Ignored if `seq_task='activity_suffix'`.
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
    best_MAE_ttne : float, optional
        Best validation Mean Absolute Error for the time till 
        next event suffix prediction. The defaults apply if the training 
        loop is initialized for the first time for a given configuration. 
        If the training loop is resumed from a certain checkpoint, the 
        best results of the previous training loop should be given. 
        When `seq_task='activity_suffix'`, the default can be retained
        as well.
    best_DL_sim : float, optional
        Best validation 1-'normalized Damerau-Levenshtein distance for 
        activity suffix prediction so far. The defaults apply if the  
        training loop is initialized for the first time for a given 
        configuration. If the training loop is resumed from a certain 
        checkpoint, the best results of the previous training loop should 
        be given. When `seq_task='timestamp_suffix'`, the default can be 
        retained as well.
    max_norm : float, optional
        Max gradient norm used for clipping during training. By default 2.
    seed : int or None, optional
        Seed for reproducibility. By default None. When an integer seed 
        is provided, it is used for shuffling and sampling the training
        instances each epoch. If None, the epoch numbers are used as
        seeds for shuffling and sampling.
    """
    if lr_scheduler_present:
        if lr_scheduler==None:
            print("No lr_scheduler provided.")
            return -1, -1, -1, -1

    # Checking whether GPU is being used
    print("Device: {}".format(device))

    # Assigning model to GPU. 
    model.to(device)
    act_sufbool = (seq_task=='activity_suffix') 
    ts_sufbool = (seq_task=='timestamp_suffix')
    if not (act_sufbool or ts_sufbool):
        raise ValueError(
            "`seq_task={}` is not a valid argument. It ".format(seq_task) + 
            "should be either `'activity_suffix'` or "
            "`'timestamp_suffix'`."
        )

    if clen_dis_ref:
        print("CaLenDiR training activated")
    else:
        print("Default training mode. CaLenDiR training not activated.")
    
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

    # Tracking average og gradient norms (before clipping if needed)
    average_og_norms = []

    # Tracking average final gradient norm (after clipping), used for 
    # updating 
    average_clipped_norms = []

    # Track evolution of validation metrics over the epoch loop by initializing empty lists. 
    if ts_sufbool: 
        avg_MAE_ttne_stand_glob, avg_MAE_ttne_minutes_glob = [], []
    
    elif act_sufbool: 
        avg_dam_lev_glob = []

    # Specifing composite loss function 
    loss_fn = SST_Loss(seq_task=seq_task, num_classes=num_classes, clen_dis_ref=clen_dis_ref)

    num_epochs_not_improved = 0
    for epoch in range(start_epoch, start_epoch + num_epochs):

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

        # Process current epoch
        model, optimizer, epoch_averages = train_epoch(model, 
                                                       train_dataloader, 
                                                       seq_task=seq_task,
                                                       optimizer=optimizer,
                                                       loss_fn=loss_fn,
                                                       epoch_number=epoch,
                                                       max_norm=max_norm)
        
        train_losses_global.append(epoch_averages[0])

        average_og_norms.append(epoch_averages[1])

        average_clipped_norms.append(epoch_averages[-1])

        # Set the model to evaluation mode and disabling dropout
        model.eval()

        if clen_dis_ref:
            # Second tuple element contains list of CB metrics
            _, inf_results, _ = inference_loop(model=model, 
                                               inference_dataset=val_dataset,
                                               seq_task=seq_task,
                                               num_categoricals_pref=num_categoricals_pref, 
                                               mean_std_ttne=mean_std_ttne, 
                                               mean_std_tsp=mean_std_tsp, 
                                               mean_std_tss=mean_std_tss, 
                                               og_caseint=og_caseint_val,
                                               results_path=None, 
                                               val_batch_size=4096)
        else: 
            # First tuple element contains list of IB (default) metrics
            inf_results, _, _ = inference_loop(model=model, 
                                               inference_dataset=val_dataset,
                                               seq_task=seq_task,
                                               num_categoricals_pref=num_categoricals_pref, 
                                               mean_std_ttne=mean_std_ttne, 
                                               mean_std_tsp=mean_std_tsp, 
                                               mean_std_tss=mean_std_tss, 
                                               og_caseint=og_caseint_val,
                                               results_path=None, 
                                               val_batch_size=4096)
            
        better = False 
        if act_sufbool:
            avg_dam_lev = inf_results[0]
            avg_dam_lev_glob.append(avg_dam_lev)
            print("Avg 1-(normalized) DL distance acitivty suffix prediction validation set: {}".format(avg_dam_lev))
            if avg_dam_lev > best_DL_sim:
                better = True 
                best_DL_sim = avg_dam_lev

        elif ts_sufbool:
            avg_MAE_ttne_stand, avg_MAE_ttne_minutes = inf_results
            avg_MAE_ttne_stand_glob.append(avg_MAE_ttne_stand)
            avg_MAE_ttne_minutes_glob.append(avg_MAE_ttne_minutes)
            print("Avg MAE TTNE prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_ttne_stand, avg_MAE_ttne_minutes))
            if avg_MAE_ttne_stand < best_MAE_ttne: 
                better = True
                best_MAE_ttne = avg_MAE_ttne_stand

        if better == False: 
            num_epochs_not_improved += 1
        else:
            num_epochs_not_improved = 0

        # Saving checkpoint every epoch
        model_path = os.path.join(path_name, 'model_epoch_{}.pt'.format(epoch))
        checkpoint = {'epoch:' : epoch, 
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(), 
                        'loss': 999}
        torch.save(checkpoint, model_path)
            
        if lr_scheduler_present:
            # Update the learning rate
            lr_scheduler.step()
            
        torch.cuda.empty_cache()


        if num_epochs_not_improved >= patience:
            print("No improvements in validation loss for {} consecutive epochs. Final epoch: {}".format(patience, epoch))
            break
    # Writing training progress to csv at the end of the current training loop
    results_path = os.path.join(path_name, 'backup_results.csv')
    epoch_list = [i for i in range(len(train_losses_global))]

    results = pd.DataFrame(data = {'epoch' : epoch_list, 
                        'training loss' : train_losses_global, 
                        'Average OG gradient norm' : average_og_norms,
                        'Average clipped gradient norm' : average_clipped_norms})
    
    if act_sufbool:
        results['Activity suffix: 1-DL (validation)'] = avg_dam_lev_glob
    elif ts_sufbool:
        results['TTNE - standardized MAE validation'] = avg_MAE_ttne_stand_glob
        results['TTNE - minutes MAE validation'] = avg_MAE_ttne_minutes_glob


    results.to_csv(results_path, index=False)