"""Pipeline for training and evaluating SuTraN+ under equal loss weighting.

This module wires together data loading, CaLenDiR sampling, SuTraN+
training, early stopping, and final inference for the equal-loss
configuration. While the default arguments correspond to the paper
setup, callers can override them to explore alternative logs or
hyperparameters.
"""
import pandas as pd 
import numpy as np 
import torch 
from torch.utils.data import TensorDataset, DataLoader
import os
import pickle 
import sys 
from Utils.callback_selection import get_target_metrics_dict, select_best_epoch

def load_checkpoint(model, path_to_checkpoint, train_or_eval, lr):
    """Load a saved SuTraN+ checkpoint together with its optimizer state.

    https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html 

    Parameters
    ----------
    model : SuTraN.SuTraN
        Freshly initialised SuTraN+ instance with the same hyperparameters
        as the saved model. The weights in `path_to_checkpoint` will be
        loaded into this instance.
    path_to_checkpoint : str
        Absolute or relative path pointing to the ``.pt`` checkpoint file.
    train_or_eval : {'train', 'eval'}
        Whether the returned model should be put back into training mode
        (for fine-tuning / resuming) or evaluation mode.
    lr : float
        Learning-rate value to use when rebuilding the AdamW optimizer
        before loading its state dict.
    
    Returns
    -------
    model : SuTraN.SuTraN
        The input model with weights restored from the checkpoint.
    optimizer : torch.optim.AdamW
        Optimizer instance with the saved state dict applied.
    final_epoch_trained : int
        Epoch index stored in the checkpoint. When resuming, continue
        from ``final_epoch_trained + 1``.
    final_loss : float
        Scalar training loss that was stored alongside the checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print(device)

    if (train_or_eval!= 'train') and (train_or_eval!= 'eval'):
        print("train_or_eval argument should be either 'train' or 'eval'.")
        return -1, -1, -1, -1

    checkpoint = torch.load(path_to_checkpoint)
    # Loading saved weights of the model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    # Loading saved state of the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    # optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Loading number of last epoch that the saved model was trained for. 
    final_epoch_trained = checkpoint['epoch:']
    # Last loss of the trained model. 
    final_loss = checkpoint['loss']

    if train_or_eval == 'train':
        model.train()
    else: 
        model.eval()
        
    return model, optimizer, final_epoch_trained, final_loss


def train_eval(log_name, 
               median_caselen,
               outcome_bool, 
               out_mask,
               clen_dis_ref=True, 
               out_type=None, 
               num_outclasses=None,
               out_string=None,
               seed=24):
    """Train and evaluate SuTraN+ under standard equal loss weighting.

    Parameters
    ----------
    log_name : str
        Name of the event log on which the model is trained. Should be 
        the same string as the one specified for the `log_name` parameter 
        of the `log_to_tensors()` function in the 
        `Preprocessing\from_log_to_tensors.py` module. 
    median_caselen : int
        Median case length original cases. 
    outcome_bool : bool
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
        produces the labels.
    out_mask : bool
        Indicates whether an instance-level outcome mask is needed 
        for preventing instances, of which the inputs (prefix events) 
        contain information directly revealing the outcome label, 
        from contributing to loss function pertaining to the outcome 
        prediction head (`True`), or not (`False`).
    clen_dis_ref : bool, optional
        If `True`, Case Length Distribution-Reflective (CaLenDiR) 
        Training is performed. This includes the application of Uniform 
        Case-Based Sampling (UCBS) of instances each epoch, and 
        Suffix-Length-Normalized Loss Functions. If `False`, the default 
        training procedure, in which all instances are used for training 
        each epoch and in which no loss function normalization is 
        performed (and hence in which case-length distortion is not 
        addressed), is performed. `True` by default. 
        Note that the default (non-CaLenDiR) training 
        procedure is performed by choosing the non-default parameter 
        value (`False`) for the boolean `clen_dis_ref` parameter. By 
        retaining the default `True` parameter value for `clen_dis_ref`, 
        the model is trained according to the non-default CaLenDiR 
        training procedure. 
    out_type : {None, 'binary_outcome', 'multiclass_outcome'}, optional
        The type of outcome prediction that is being performed in the 
        multi-task setting. Only taken into account of outcome prediction 
        is included in the event log to begin with, and hence if 
        `outcome_bool=True`. If so, `'binary_outcome'` denotes binary 
        outcome (BO) prediction (binary classification), while 
        `'multiclass_outcome'` denotes multi-class outcome (MCO) 
        prediction (Multi-Class classification). 
    num_outclasses : int or None, optional
        The number of outcome classes in case 
        `outcome_bool=True` and `out_type='multiclass_outcome'`. By 
        default `None`. 
    out_string : str or None, optional
        An optional string to specify the outcome, in case the event log 
        trained and evaluated for (specified by `log_name`) contains 
        multiple potential outcomes. This will be incorporated in 
        the path to which the results are stored. 
    seed : int, optional
        Seed value to set for reproducibility. By default 24.
    """
    data_path = log_name

    storage_path = log_name

    def load_dict(path_name):
        with open(path_name, 'rb') as file:
            loaded_dict = pickle.load(file)
        
        return loaded_dict

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
            if not isinstance(num_outclasses, int): 
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

    # -----------------
    temp_string = log_name + '_cardin_dict.pkl'
    temp_path = os.path.join(data_path, temp_string)
    cardinality_dict = load_dict(temp_path)
    num_activities = cardinality_dict['concept:name'] + 2
    print(num_activities)

    # cardinality list prefix categoricals 
    temp_string = log_name + '_cardin_list_prefix.pkl'
    temp_path = os.path.join(data_path, temp_string)
    cardinality_list_prefix = load_dict(temp_path)

    temp_string = log_name + '_cardin_list_suffix.pkl'
    temp_path = os.path.join(data_path, temp_string)

    temp_string = log_name + '_num_cols_dict.pkl'
    temp_path = os.path.join(data_path, temp_string)
    # To retrieve the number of numerical featrues in the prefix and suffix events respectively 
    num_cols_dict = load_dict(temp_path)

    temp_string = log_name + '_cat_cols_dict.pkl'
    temp_path = os.path.join(data_path, temp_string)
    cat_cols_dict = load_dict(temp_path)

    temp_string = log_name + '_train_means_dict.pkl'
    temp_path = os.path.join(data_path, temp_string)
    train_means_dict = load_dict(temp_path)

    temp_string = log_name + '_train_std_dict.pkl'
    temp_path = os.path.join(data_path, temp_string)

    train_std_dict = load_dict(temp_path)

    mean_std_ttne = [train_means_dict['timeLabel_df'][0], train_std_dict['timeLabel_df'][0]]
    mean_std_tsp = [train_means_dict['suffix_df'][1], train_std_dict['suffix_df'][1]]
    mean_std_tss = [train_means_dict['suffix_df'][0], train_std_dict['suffix_df'][0]]
    mean_std_rrt = [train_means_dict['timeLabel_df'][1], train_std_dict['timeLabel_df'][1]]
    num_numericals_pref = len(num_cols_dict['prefix_df'])

    num_categoricals_pref = len(cat_cols_dict['prefix_df'])

    # Loading train, validation and test sets 
    # Training set 
    temp_path = os.path.join(data_path, 'train_tensordataset.pt')
    train_dataset = torch.load(temp_path)

    # Validation set
    temp_path = os.path.join(data_path, 'val_tensordataset.pt')
    val_dataset = torch.load(temp_path)

    # Test set 
    temp_path = os.path.join(data_path, 'test_tensordataset.pt')
    test_dataset = torch.load(temp_path)


    temp_path = os.path.join(data_path, 'og_caseint_train.pt')
    og_caseint_train = torch.load(temp_path)

    temp_path = os.path.join(data_path, 'og_caseint_val.pt')
    og_caseint_val = torch.load(temp_path)

    temp_path = os.path.join(data_path, 'og_caseint_test.pt')
    og_caseint_test = torch.load(temp_path)

    if outcome_bool and out_mask:
        # Loading outcome mask tensors 
        temp_path = os.path.join(data_path, 'instance_mask_out_train.pt')
        instance_mask_out_train = torch.load(temp_path)

        temp_path = os.path.join(data_path, 'instance_mask_out_val.pt')
        instance_mask_out_val = torch.load(temp_path)

        temp_path = os.path.join(data_path, 'instance_mask_out_test.pt')
        instance_mask_out_test = torch.load(temp_path)

    else: 
        instance_mask_out_train = None
        instance_mask_out_val = None 
        instance_mask_out_test = None

        # For safety: out_mask cannot be True in this case
        out_mask=False 

    # Fixed variables 
    d_model = 32 
    num_prefix_encoder_layers = 4
    num_decoder_layers = 4
    num_heads = 8 
    layernorm_embeds = True
    remaining_runtime_head = True 
    # Creating auxiliary bools 
    only_rrt = (not outcome_bool) & remaining_runtime_head
    only_out = outcome_bool & (not remaining_runtime_head)
    both = outcome_bool & remaining_runtime_head

    dropout = 0.2
    batch_size = 128

    # specifying path results and callbacks 
    model_string = 'SUTRAN_DA_results'
    if out_type: 
        model_string += '_' + out_type
        if out_string:
            model_string += '_' + out_string
    model_string += '_seed_{}'.format(seed)
    subfolder_path = os.path.join(storage_path, model_string)
    os.makedirs(subfolder_path, exist_ok=True)

    
    if clen_dis_ref: 
        backup_path = os.path.join(subfolder_path, "CaLenDiR_training")
    else:
        backup_path = os.path.join(subfolder_path, "default_training")
    
    os.makedirs(backup_path, exist_ok=True)

    backup_path = os.path.join(backup_path, "Default_Equal_Weighting")
    os.makedirs(backup_path, exist_ok=True)

    # Setting up GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)


    # Initializing model 
    import random
    # Set a seed value
    seed_value = seed

    # Set Python random seed
    random.seed(seed_value)

    # Set NumPy random seed
    np.random.seed(seed_value)

    # Set PyTorch random seed
    torch.manual_seed(seed_value)

    # If you are using CUDA (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
        # Additional settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    from SuTraN.SuTraN import SuTraN
    model = SuTraN(num_activities=num_activities, 
                        d_model=d_model, 
                        cardinality_categoricals_pref=cardinality_list_prefix, 
                        num_numericals_pref=num_numericals_pref, 
                        num_prefix_encoder_layers=num_prefix_encoder_layers, 
                        num_decoder_layers=num_decoder_layers, 
                        num_heads= num_heads, 
                        d_ff = 4*d_model, 
                        dropout=dropout, 
                        remaining_runtime_head=True, # Always included. 
                        layernorm_embeds=layernorm_embeds, 
                        outcome_bool=outcome_bool, 
                        out_type=out_type, 
                        num_outclasses=num_outclasses)

    # Assign to GPU 
    model.to(device)

    # Initializing optimizer and learning rate scheduler 
    decay_factor = 0.96
    lr = 0.0002
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_factor)

    
    # Training procedure 
    from SuTraN.train_procedure import train_model
    start_epoch = 0
    num_epochs = 200 
    num_classes = num_activities 
    batch_interval = 800
    train_model(model, 
                optimizer, 
                train_dataset, 
                val_dataset, 
                start_epoch, 
                num_epochs, 
                remaining_runtime_head,
                outcome_bool,
                num_classes, 
                batch_interval, 
                backup_path, 
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
                out_mask, 
                instance_mask_out_train, 
                instance_mask_out_val,
                out_type=out_type, 
                num_outclasses=num_outclasses,
                patience=24,
                lr_scheduler_present=True, 
                lr_scheduler=lr_scheduler, 
                seed=seed_value)
    
    # Re-initializing new model after training to load best callback
    model = SuTraN(num_activities=num_activities, 
                        d_model=d_model, 
                        cardinality_categoricals_pref=cardinality_list_prefix, 
                        num_numericals_pref=num_numericals_pref, 
                        num_prefix_encoder_layers=num_prefix_encoder_layers, 
                        num_decoder_layers=num_decoder_layers, 
                        num_heads= num_heads, 
                        d_ff = 4*d_model, 
                        dropout=dropout, 
                        remaining_runtime_head=True, # Always included. 
                        layernorm_embeds=layernorm_embeds, 
                        outcome_bool=outcome_bool, 
                        out_type=out_type, 
                        num_outclasses=num_outclasses)

    # Assign to GPU 
    model.to(device)

    # Specifying path of csv in which the training and validation results 
    # of every epoch are stored. 
    final_results_path = os.path.join(backup_path, 'backup_results.csv')

    # Determining best epoch based on the validation 
    # scores for RRT and Activity Suffix prediction
    df = pd.read_csv(final_results_path)

    task_list = ['activity_suffix', 'timestamp_suffix']

    if remaining_runtime_head: 
        task_list.append('remaining_runtime')
    
    if bin_outbool: 
        task_list.append('binary_outcome')
    
    if multic_outbool:
        task_list.append('multiclass_outcome')

    target_metrics = get_target_metrics_dict(task_list)

    # Determine epoch number best epcoh, and the corresponding row 
    # in the backup results df 
    best_epoch, best_row = select_best_epoch(df, target_metrics)

    best_epoch = int(best_epoch)

    # The models are stored with the string underneath
    best_epoch_string = 'model_epoch_{}.pt'.format(best_epoch)
    best_epoch_path = os.path.join(backup_path, best_epoch_string)

    # Load best model into memory again 
    model, _, _, _ = load_checkpoint(model, path_to_checkpoint=best_epoch_path, train_or_eval='eval', lr=0.002)
    model.to(device)
    model.eval()

    # Running final inference on test set 
    from SuTraN.inference_procedure import inference_loop

    # Initializing directory for final test set results 
    results_path = os.path.join(backup_path, "TEST_SET_RESULTS")
    os.makedirs(results_path, exist_ok=True)


    inf_results_IB, inf_results_CB, pref_suf_results = inference_loop(model=model, 
                                                                      inference_dataset=test_dataset,
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
                                                                      og_caseint=og_caseint_test,
                                                                      instance_mask_out=instance_mask_out_test,
                                                                      results_path=results_path, 
                                                                      val_batch_size=2048)

    #######################################################
    ###########   INSTANCE-BASED (IB) METRICS   ###########
    #######################################################

    avg_MAE_ttne_stand_IB, avg_MAE_ttne_minutes_IB = inf_results_IB[:2]
    # Average Normalized Damerau-Levenshtein similarity Activity Suffix 
    # prediction
    avg_dam_lev_IB = inf_results_IB[2]
    if only_rrt:
        # MAE standardized RRT predictions 
        avg_MAE_stand_RRT_IB = inf_results_IB[3]
        # MAE RRT converted to minutes
        avg_MAE_minutes_RRT_IB = inf_results_IB[4]

    elif only_out:
        if bin_outbool: # binary outcome prediction metrics
            # Binary Cross Entropy outcome prediction
            avg_BCE_out_IB = inf_results_IB[3]
            # AUC-ROC outcome prediction
            auc_roc_IB = inf_results_IB[4]
            # AUC-PR outcome prediction
            auc_pr_IB = inf_results_IB[5]
            
            binary_dict_IB = inf_results_IB[6]
        
        elif multic_outbool: 
            # Categorical Cross Entropy Inf set MCO prediction
            avg_CE_MCO_IB = inf_results_IB[3]
            # Dict of additional validation metrics 
            mc_dict_IB = inf_results_IB[4]

    elif both: 
        # MAE standardized RRT predictions 
        avg_MAE_stand_RRT_IB = inf_results_IB[3]
        # MAE RRT converted to minutes
        avg_MAE_minutes_RRT_IB = inf_results_IB[4]

        if bin_outbool: # binary outcome prediction metrics
            # Binary Cross Entropy outcome prediction
            avg_BCE_out_IB = inf_results_IB[5]
            # AUC-ROC outcome prediction
            auc_roc_IB = inf_results_IB[6]
            # AUC-PR outcome prediction
            auc_pr_IB = inf_results_IB[7]
            
            binary_dict_IB = inf_results_IB[8]

        
        elif multic_outbool: # Multi-Class Outcome (MCO) prediction metrics 
            # Categorical Cross Entropy Inf set MCO prediction
            avg_CE_MCO_IB = inf_results_IB[5]
            # Dict of additional validation metrics 
            mc_dict_IB = inf_results_IB[6]

    # Retrieving and storing dictionary of the metrics averaged over all 
    # test set instances (prefix-suffix pairs)
    avg_results_dict_IB = {"MAE TTNE minutes" : avg_MAE_ttne_minutes_IB, 
                           "DL sim" : avg_dam_lev_IB}
    
    # Printing INSTANCE-BASED averaged results 
    print("INSTANCE-BASED (IB) METRICS:")
    print("IB - Avg MAE TTNE prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_ttne_stand_IB, avg_MAE_ttne_minutes_IB))
    print("IB - Avg 1-(normalized) DL distance acitivty suffix prediction validation set: {}".format(avg_dam_lev_IB))

    if remaining_runtime_head: 
        print("IB - Avg MAE RRT prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_stand_RRT_IB, avg_MAE_minutes_RRT_IB))
        avg_results_dict_IB['MAE RRT minutes'] = avg_MAE_minutes_RRT_IB

    if bin_outbool: # Binary Outcome Prediction
        acc_IB = binary_dict_IB['accuracy']

        f1_IB = binary_dict_IB['f1']

        precision_IB = binary_dict_IB['precision']

        recall_IB = binary_dict_IB['recall']

        balanced_accuracy_IB = binary_dict_IB['balanced_accuracy']

        print("IB - Avg BCE outcome prediction validation set: {}".format(avg_BCE_out_IB))
        print("IB - AUC-ROC outcome prediction validation set: {}".format(auc_roc_IB))
        print("IB - AUC-PR outcome prediction validation set: {}".format(auc_pr_IB))
        print("IB - Accuracy outcome prediction validation set: {}".format(acc_IB))
        print("IB - F1 score outcome prediction validation set: {}".format(f1_IB))
        print("IB - Precision outcome prediction validation set: {}".format(precision_IB))
        print("IB - Recall outcome prediction validation set: {}".format(recall_IB))
        print("IB - Balanced Accuracy outcome prediction validation set: {}".format(balanced_accuracy_IB))


        print("IB - Avg BCE outcome prediction validation set: {}".format(avg_BCE_out_IB))
        print("IB - AUC-ROC outcome prediction validation set: {}".format(auc_roc_IB))
        print("IB - AUC-PR outcome prediction validation set: {}".format(auc_pr_IB))
        
        avg_results_dict_IB['BCE'] = avg_BCE_out_IB
        avg_results_dict_IB["AUC-ROC"] = auc_roc_IB
        avg_results_dict_IB["AUC-PR"] = auc_pr_IB
        avg_results_dict_IB["Binary Accuracy (th 0.5)"] = acc_IB
        avg_results_dict_IB["Binary F1 (th 0.5)"] = f1_IB
        avg_results_dict_IB["Binary Precision (th 0.5)"] = precision_IB
        avg_results_dict_IB["Binary Recall (th 0.5)"] = recall_IB
        avg_results_dict_IB["Binary Bal. Accuracy (th 0.5)"] = balanced_accuracy_IB

    if multic_outbool: # Multi-Class Outcome prediction
        acc_IB = mc_dict_IB['accuracy']

        macro_f1_IB = mc_dict_IB['macro_f1']

        weighted_f1_IB = mc_dict_IB['weighted_f1']

        macro_precision_IB = mc_dict_IB['macro_precision']

        weighted_precision_IB = mc_dict_IB['weighted_precision']

        macro_recall_IB = mc_dict_IB['macro_recall']

        weighted_recall_IB = mc_dict_IB['weighted_recall']

        print("IB - Avg CE Multi-Class Outcome (MCO) prediction validation set: {}".format(avg_CE_MCO_IB))
        print("IB - Accuracy MCO prediction validation set: {}".format(acc_IB))
        print("IB - Macro F1 MCO prediction validation set: {}".format(macro_f1_IB))
        print("IB - Weighted F1 MCO prediction validation set: {}".format(weighted_f1_IB))
        print("IB - Macro Precision MCO prediction validation set: {}".format(macro_precision_IB))
        print("IB - Weighted Precision MCO prediction validation set: {}".format(weighted_precision_IB))
        print("IB - Macro Recall MCO prediction validation set: {}".format(macro_recall_IB))
        print("IB - Weighted Recall MCO prediction validation set: {}".format(weighted_recall_IB))

        avg_results_dict_IB["CE"] = avg_CE_MCO_IB
        avg_results_dict_IB["Multi-Class Accuracy"] = acc_IB
        avg_results_dict_IB["Macro-F1"] = macro_f1_IB
        avg_results_dict_IB["Weighted-F1"] = weighted_f1_IB
        avg_results_dict_IB["Macro-Precision"] = macro_precision_IB
        avg_results_dict_IB["Weighted-Precision"] = weighted_precision_IB
        avg_results_dict_IB["Macro-Recall"] = macro_recall_IB
        avg_results_dict_IB["Weighted-Recall"] = weighted_recall_IB


    path_name_average_results_IB = os.path.join(results_path, 'averaged_results_IB.pkl')


    #######################################################
    #############   CASE-BASED (CB) METRICS   #############
    #######################################################

    avg_MAE_ttne_stand_CB, avg_MAE_ttne_minutes_CB = inf_results_CB[:2]
    # Average Normalized Damerau-Levenshtein similarity Activity Suffix 
    # prediction
    avg_dam_lev_CB = inf_results_CB[2]
    if only_rrt:
        # MAE standardized RRT predictions 
        avg_MAE_stand_RRT_CB = inf_results_CB[3]
        # MAE RRT converted to minutes
        avg_MAE_minutes_RRT_CB = inf_results_CB[4]

    elif only_out:
        if bin_outbool: # binary outcome prediction metrics
            # Binary Cross Entropy outcome prediction
            avg_BCE_out_CB = inf_results_CB[3]
            # AUC-ROC outcome prediction
            auc_roc_CB = inf_results_CB[4]
            # AUC-PR outcome prediction
            auc_pr_CB = inf_results_CB[5]
            
            binary_dict_CB = inf_results_CB[6]
        
        elif multic_outbool: 
            # Categorical Cross Entropy Inf set MCO prediction
            avg_CE_MCO_CB = inf_results_CB[3]
            # Dict of additional validation metrics 
            mc_dict_CB = inf_results_CB[4]

    elif both: 
        # MAE standardized RRT predictions 
        avg_MAE_stand_RRT_CB = inf_results_CB[3]
        # MAE RRT converted to minutes
        avg_MAE_minutes_RRT_CB = inf_results_CB[4]

        if bin_outbool: # binary outcome prediction metrics
            # Binary Cross Entropy outcome prediction
            avg_BCE_out_CB = inf_results_CB[5]
            # AUC-ROC outcome prediction
            auc_roc_CB = inf_results_CB[6]
            # AUC-PR outcome prediction
            auc_pr_CB = inf_results_CB[7]
            
            binary_dict_CB = inf_results_CB[8]

        
        elif multic_outbool: # Multi-Class Outcome (MCO) prediction metrics 
            # Categorical Cross Entropy Inf set MCO prediction
            avg_CE_MCO_CB = inf_results_CB[5]
            # Dict of additional validation metrics 
            mc_dict_CB = inf_results_CB[6]

    # Retrieving and storing dictionary of the metrics averaged over all 
    # test set instances (prefix-suffix pairs)
    avg_results_dict_CB = {"MAE TTNE minutes" : avg_MAE_ttne_minutes_CB, 
                           "DL sim" : avg_dam_lev_CB}
    
    # Printing CASE-BASED averaged results 
    print("CASE-BASED (CB) METRICS:")
    print("CB - Avg MAE TTNE prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_ttne_stand_CB, avg_MAE_ttne_minutes_CB))
    print("CB - Avg 1-(normalized) DL distance acitivty suffix prediction validation set: {}".format(avg_dam_lev_CB))

    if remaining_runtime_head: 
        print("CB - Avg MAE RRT prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_stand_RRT_CB, avg_MAE_minutes_RRT_CB))
        avg_results_dict_CB['MAE RRT minutes'] = avg_MAE_minutes_RRT_CB

    if bin_outbool: # Binary Outcome Prediction
        acc_CB = binary_dict_CB['accuracy']

        f1_CB = binary_dict_CB['f1']

        precision_CB = binary_dict_CB['precision']

        recall_CB = binary_dict_CB['recall']

        balanced_accuracy_CB = binary_dict_CB['balanced_accuracy']

        print("CB - Avg BCE outcome prediction validation set: {}".format(avg_BCE_out_CB))
        print("CB - AUC-ROC outcome prediction validation set: {}".format(auc_roc_CB))
        print("CB - AUC-PR outcome prediction validation set: {}".format(auc_pr_CB))
        print("CB - Accuracy outcome prediction validation set: {}".format(acc_CB))
        print("CB - F1 score outcome prediction validation set: {}".format(f1_CB))
        print("CB - Precision outcome prediction validation set: {}".format(precision_CB))
        print("CB - Recall outcome prediction validation set: {}".format(recall_CB))
        print("CB - Balanced Accuracy outcome prediction validation set: {}".format(balanced_accuracy_CB))


        print("CB - Avg BCE outcome prediction validation set: {}".format(avg_BCE_out_CB))
        print("CB - AUC-ROC outcome prediction validation set: {}".format(auc_roc_CB))
        print("CB - AUC-PR outcome prediction validation set: {}".format(auc_pr_CB))
        
        avg_results_dict_CB['BCE'] = avg_BCE_out_CB
        avg_results_dict_CB["AUC-ROC"] = auc_roc_CB
        avg_results_dict_CB["AUC-PR"] = auc_pr_CB
        avg_results_dict_CB["Binary Accuracy (th 0.5)"] = acc_CB
        avg_results_dict_CB["Binary F1 (th 0.5)"] = f1_CB
        avg_results_dict_CB["Binary Precision (th 0.5)"] = precision_CB
        avg_results_dict_CB["Binary Recall (th 0.5)"] = recall_CB
        avg_results_dict_CB["Binary Bal. Accuracy (th 0.5)"] = balanced_accuracy_CB

    if multic_outbool: # Multi-Class Outcome prediction
        acc_CB = mc_dict_CB['accuracy']

        macro_f1_CB = mc_dict_CB['macro_f1']

        weighted_f1_CB = mc_dict_CB['weighted_f1']

        macro_precision_CB = mc_dict_CB['macro_precision']

        weighted_precision_CB = mc_dict_CB['weighted_precision']

        macro_recall_CB = mc_dict_CB['macro_recall']

        weighted_recall_CB = mc_dict_CB['weighted_recall']

        print("CB - Avg CE Multi-Class Outcome (MCO) prediction validation set: {}".format(avg_CE_MCO_CB))
        print("CB - Accuracy MCO prediction validation set: {}".format(acc_CB))
        print("CB - Macro F1 MCO prediction validation set: {}".format(macro_f1_CB))
        print("CB - Weighted F1 MCO prediction validation set: {}".format(weighted_f1_CB))
        print("CB - Macro Precision MCO prediction validation set: {}".format(macro_precision_CB))
        print("CB - Weighted Precision MCO prediction validation set: {}".format(weighted_precision_CB))
        print("CB - Macro Recall MCO prediction validation set: {}".format(macro_recall_CB))
        print("CB - Weighted Recall MCO prediction validation set: {}".format(weighted_recall_CB))

        avg_results_dict_CB["CE"] = avg_CE_MCO_CB
        avg_results_dict_CB["Multi-Class Accuracy"] = acc_CB
        avg_results_dict_CB["Macro-F1"] = macro_f1_CB
        avg_results_dict_CB["Weighted-F1"] = weighted_f1_CB
        avg_results_dict_CB["Macro-Precision"] = macro_precision_CB
        avg_results_dict_CB["Weighted-Precision"] = weighted_precision_CB
        avg_results_dict_CB["Macro-Recall"] = macro_recall_CB
        avg_results_dict_CB["Weighted-Recall"] = weighted_recall_CB


    path_name_average_results_CB = os.path.join(results_path, 'averaged_results_CB.pkl')


    
    # Retrieving and storing the dictionaries with the 
    # averaged results per prefix and suffix length
    results_dict_pref = pref_suf_results[0]
    results_dict_suf = pref_suf_results[1]


    path_name_prefix = os.path.join(results_path, 'prefix_length_results_dict.pkl')
    path_name_suffix = os.path.join(results_path, 'suffix_length_results_dict.pkl')
    with open(path_name_prefix, 'wb') as file:
        pickle.dump(results_dict_pref, file)
    with open(path_name_suffix, 'wb') as file:
        pickle.dump(results_dict_suf, file)
    with open(path_name_average_results_IB, 'wb') as file:
        pickle.dump(avg_results_dict_IB, file)
    with open(path_name_average_results_CB, 'wb') as file:
        pickle.dump(avg_results_dict_CB, file)




if __name__ == "__main__":
    import argparse

    # Define a helper to interpret boolean strings.
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # Define a helper for parameters that could be an int or None.
    def str2int_or_none(v):
        if v.lower() == "none":
            return None
        try:
            return int(v)
        except ValueError:
            raise argparse.ArgumentTypeError("Expected an integer or 'None'.")

    parser = argparse.ArgumentParser(
        description="Train and evaluate SuTraN+ using the Equal Weighting setup."
    )
    parser.add_argument("--log_name", type=str, required=True,
                        help="Dataset name (e.g., 'BPIC_17').")
    parser.add_argument("--median_caselen", type=int, required=True,
                        help="Median case length.")
    parser.add_argument("--outcome_bool", type=str2bool, required=True,
                        help="Include outcome prediction (True/False).")
    parser.add_argument("--out_mask", type=str2bool, required=True,
                        help="Use outcome mask (True/False).")
    parser.add_argument("--clen_dis_ref", type=str2bool, required=True,
                        help="Use CaLenDiR training (True/False).")
    # Accept out_type as a string that can be "binary_outcome", "multiclass_outcome", or "None"
    parser.add_argument("--out_type", type=str, required=True,
                        help="Outcome type ('binary_outcome', 'multiclass_outcome', or 'None').")
    # Accept num_outclasses as a string that can be an integer (e.g., '3') or 'None'
    parser.add_argument("--num_outclasses", type=str, required=True,
                        help="Number of outcome classes (e.g., '3' for multiclass) or 'None'.")
    parser.add_argument("--out_string", type=str, default="",
                        help="Optional outcome string.")
    parser.add_argument("--seed", type=int, default=24,
                        help="Seed value for reproducibility.")

    args = parser.parse_args()

    # Convert the string parameters for out_type and num_outclasses.
    if args.out_type.lower() == "none":
        args.out_type = None
    # Convert num_outclasses to int or None
    args.num_outclasses = str2int_or_none(args.num_outclasses)

    # Call the train_eval function with constant parameters and the parsed hyperparameters.
    train_eval(
        log_name=args.log_name,
        median_caselen=args.median_caselen,
        outcome_bool=args.outcome_bool,
        out_mask=args.out_mask,
        clen_dis_ref=args.clen_dis_ref,
        out_type=args.out_type,
        num_outclasses=args.num_outclasses,
        out_string=args.out_string if args.out_string != "" else None,
        seed=args.seed
    )
