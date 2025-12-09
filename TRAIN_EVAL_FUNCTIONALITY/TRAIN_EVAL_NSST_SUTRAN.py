"""Module containing entire train and evaluation pipeline for the 
Non-Sequential Single-Task (NSST) version of SuTraN, catering to the 
following three Non-Sequantial prediction tasks: 

1. Remaining Runtime (RRT) prediction (`scalar_task='remaining_runtime'`) 
2. Binary Outcome (BO) prediction (`scalar_task='binary_outcome'`) 
3. Multi-Class Outcome (MCO) prediction 
   (`scalar_task='multiclass_outcome'`)
"""
import pandas as pd 
import numpy as np 
import torch 
from torch.utils.data import TensorDataset, DataLoader
import os
import pickle 

# import model 
from NSST_SuTraN.SuTraN_NSST import SuTraN_NSST

from NSST_SuTraN.train_procedure import train_model
from NSST_SuTraN.inference_procedure import inference_loop

# import preprocessing utils 
from NSST_SuTraN.convert_tensordata import add_prefix_CLStoken, subset_data

# import functionality automatic epoch callback selection 
# after training (based on validation scores)
from Utils.callback_selection import get_target_metrics_dict, select_best_epoch

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
    # final_epoch_trained = checkpoint["epoch"]
    final_epoch_trained = checkpoint['epoch:']

    # Set the model to train or eval mode
    if train_or_eval == "train":
        model.train()
    else:
        model.eval()

    return model, optimizer, final_epoch_trained


def train_eval(log_name, 
               median_caselen, 
               scalar_task, 
               out_mask, 
               final_embedding='last', 
               lr=0.0002,
               clen_dis_ref=True, 
               num_outclasses=None,
               out_string=None,
               seed=24): 
    """_summary_

    Parameters
    ----------
    log_name : str
        Name of the event log on which the model is trained. Should be 
        the same string as the one specified for the `log_name` parameter 
        of the `log_to_tensors()` function in the 
        `Preprocessing\from_log_to_tensors.py` module. 
    median_caselen : int
        Median number of events per original case in the training log. 
        Used by the CaLenDiR sampling routine to anchor 
        case-length-aware subsampling.
    scalar_task : {'remaining_runtime', 'binary_outcome', 'multiclass_outcome'}
        The scalar prediction task trained and evaluated for. Either 
        `'remaining_runtime'` `'binary_outcome'` or 
        `'multiclass_outcome'`.
    out_mask : bool
        If `True`, the data includes an outcome mask tensor to suppress 
        instances whose prefixes already reveal the label from 
        contributing to the outcome loss/metrics.
    final_embedding : {'CLS', 'last'}, optional
        Indicates which encoder embedding feeds the scalar head:
        `'CLS'` consumes an explicit CLS token (if prepended in the
        data), whereas `'last'` uses the final non-padded prefix
        event. This switch was kept for early ablation purposes;
        preliminary trials showed `'last'` to be consistently stronger,
        so all paper results fix this parameter to `'last'`. Default
        is `'last'`.
    lr : float, optional 
        Learning rate applied during training, by default 0.0002. 
    clen_dis_ref : bool, optional
        If `True`, Case Length Distribution-Reflective (CaLenDiR) 
        Training is performed. This includes the application of Uniform 
        Case-Based Sampling (UCBS) of instances each epoch, and 
        Suffix-Length-Normalized Loss Functions. If `False`, the default 
        training procedure, in which all instances are used for training 
        each epoch and in which no loss function normalization is 
        performed (and hence in which case-length distortion is not 
        addressed), is performed. By default `True`.
    num_outclasses : int or None, optional
        The number of outcome classes in case 
        `scalar_task='multiclass_outcome'`. By default `None`. 
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
    
    outcome_bool = (scalar_task=='binary_outcome') or (scalar_task=='multiclass_outcome')
    # binary out bool
    bin_outbool = (scalar_task=='binary_outcome')
    # multiclass out bool
    multic_outbool = (scalar_task=='multiclass_outcome')

    if multic_outbool: 
        if num_outclasses is None: 
            raise ValueError(
                "When `scalar_task='multiclass_outcome'`, "
                "'num_outclasses' should be given an integer argument."
            )

    # Reading in the data 
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
    num_heads = 8 
    layernorm_embeds = True

    dropout = 0.2
    batch_size = 128


    # specifying path results and callbacks 
    model_string = 'SuTraN_NSST_'
    model_string += scalar_task 
    if out_string: 
        model_string += '_' + out_string
    model_string += '_' + final_embedding
    model_string += '_' + str(lr)
    model_string += '_seed_{}'.format(seed)


    subfolder_path = os.path.join(storage_path, model_string)
    os.makedirs(subfolder_path, exist_ok=True)

    if clen_dis_ref: 
        backup_path = os.path.join(subfolder_path, "CaLenDiR_training")
    else:
        backup_path = os.path.join(subfolder_path, "default_training")

    # One level less for this one. Only one training option. 
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
    
    model = SuTraN_NSST(num_activities=num_activities, 
                        d_model=d_model, 
                        cardinality_categoricals_pref=cardinality_list_prefix, 
                        num_numericals_pref=num_numericals_pref, 
                        scalar_task=scalar_task, 
                        num_outclasses=num_outclasses, 
                        num_prefix_encoder_layers=num_prefix_encoder_layers, 
                        num_heads=num_heads, 
                        d_ff=4*d_model,
                        dropout=dropout, 
                        final_embedding=final_embedding, 
                        layernorm_embeds=layernorm_embeds)

    # Assign to GPU 
    model.to(device)

    # Initializing optimizer and learning rate scheduler 
    decay_factor = 0.96
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_factor)

    # Subsetting the train, validation and test sets, originally 
    # constructed for the Multi-Task version of SuTraN, for the 
    # NSST variant of SuTraN, given the specified prediction task 
    train_dataset = subset_data(train_dataset, num_categoricals_pref, scalar_task)
    val_dataset = subset_data(val_dataset, num_categoricals_pref, scalar_task)
    test_dataset = subset_data(test_dataset, num_categoricals_pref, scalar_task)

    if final_embedding=='CLS': 
        # Add a CLS token to all prefix tensors 
        train_dataset = add_prefix_CLStoken(train_dataset, 
                                            cardinality_list_prefix, 
                                            num_numericals_pref)
        
        val_dataset = add_prefix_CLStoken(val_dataset, 
                                          cardinality_list_prefix, 
                                          num_numericals_pref)
        
        test_dataset = add_prefix_CLStoken(test_dataset, 
                                           cardinality_list_prefix, 
                                           num_numericals_pref)
        

    # Training procedure 
    start_epoch = 0
    num_epochs = 200 
    num_classes = num_activities 
    batch_interval = 3000

    train_model(model=model, 
                optimizer=optimizer, 
                train_dataset=train_dataset, 
                val_dataset=val_dataset, 
                start_epoch=start_epoch, 
                num_epochs=num_epochs, 
                scalar_task=scalar_task, 
                num_classes=num_classes, 
                batch_interval=batch_interval, 
                path_name=backup_path, 
                num_categoricals_pref=num_categoricals_pref, 
                batch_size=batch_size, 
                clen_dis_ref=clen_dis_ref, 
                og_caseint_train=og_caseint_train, 
                og_caseint_val=og_caseint_val,
                median_caselen=median_caselen,
                mean_std_rrt=mean_std_rrt, # actually only needed in case of 
                                           # scalar_task='remaining_runtime'
                out_mask=out_mask, 
                instance_mask_out_train=instance_mask_out_train, 
                instance_mask_out_val=instance_mask_out_val, 
                num_outclasses=num_outclasses, # only needed in case of ... 
                patience=24, 
                lr_scheduler_present=True, 
                lr_scheduler=lr_scheduler, 
                seed=seed_value)

    # Re-initializing new model after training to load best callback
    model = SuTraN_NSST(num_activities=num_activities, 
                        d_model=d_model, 
                        cardinality_categoricals_pref=cardinality_list_prefix, 
                        num_numericals_pref=num_numericals_pref, 
                        scalar_task=scalar_task, 
                        num_outclasses=num_outclasses, 
                        num_prefix_encoder_layers=num_prefix_encoder_layers, 
                        num_heads=num_heads, 
                        d_ff=4*d_model,
                        dropout=dropout, 
                        final_embedding=final_embedding, 
                        layernorm_embeds=layernorm_embeds)
    # Assign to GPU 
    model.to(device)

    # Specifying path of csv in which the training and validation results 
    # of every epoch are stored. 
    final_results_path = os.path.join(backup_path, 'backup_results.csv')

    # Determining best epoch based on the validation scores of the respective 
    # task at hand 
    df = pd.read_csv(final_results_path)

    # Derive appropriate target_metrics dict for 
    # automatic epoch callback selection
    target_metrics = get_target_metrics_dict([scalar_task])

    # Determine number best epoch, and the corresponding row 
    # in the backup results df. 
    best_epoch, best_row = select_best_epoch(df, target_metrics)

    best_epoch = int(best_epoch)

    # The models are stored with the string underneath
    best_epoch_string = 'model_epoch_{}.pt'.format(best_epoch)
    best_epoch_path = os.path.join(backup_path, best_epoch_string)


    # Load best model into memory again 
    model, _, _ = load_checkpoint(model, path_to_checkpoint=best_epoch_path, train_or_eval='eval', lr=0.002)
    model.to(device)
    model.eval()


    # ------------------------
    # FINAL TEST SET INFERENCE
    # ------------------------
    # Initializing directory for final test set results 
    results_path = os.path.join(backup_path, "TEST_SET_RESULTS")
    os.makedirs(results_path, exist_ok=True)
    
    inf_results_IB, inf_results_CB = inference_loop(model=model, 
                                                    inference_dataset=test_dataset,
                                                    scalar_task=scalar_task, 
                                                    out_mask=out_mask,
                                                    mean_std_rrt=mean_std_rrt, 
                                                    og_caseint=og_caseint_test,
                                                    instance_mask_out=instance_mask_out_test,
                                                    num_outclasses=num_outclasses, 
                                                    results_path=results_path, 
                                                    val_batch_size=4096)
    


    # :::::::::::::::::::::::::::::::::::::::::

    print("=============================================")
    print("Final test set evaluation")
    print("=============================================")

    print(":::::::::::::::")
    print("INSTANCE-BASED (IB) METRICS:")
    print(":::::::::::::::")

    # Storing average test metrics 
    if not outcome_bool: # RRT prediction
        avg_MAE_stand_RRT_IB, avg_MAE_minutes_RRT_IB = inf_results_IB
        print("Avg MAE RRT prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_stand_RRT_IB, avg_MAE_minutes_RRT_IB))

        # Retrieving and storing dictionary of the IB metrics averaged 
        # over all test set instances
        avg_results_dict_IB = {"MAE RRT standardized" : avg_MAE_stand_RRT_IB, 
                               "MAE RRT minutes" : avg_MAE_minutes_RRT_IB}
        

    elif bin_outbool: 
        avg_BCE_out_IB, auc_roc_IB, auc_pr_IB, binary_dict_IB = inf_results_IB

        acc_IB = binary_dict_IB['accuracy']

        f1_IB = binary_dict_IB['f1']

        precision_IB = binary_dict_IB['precision']

        recall_IB = binary_dict_IB['recall']

        balanced_accuracy_IB  = binary_dict_IB['balanced_accuracy']

        print("Avg BCE outcome prediction validation set: {}".format(avg_BCE_out_IB))
        print("AUC-ROC outcome prediction validation set: {}".format(auc_roc_IB))
        print("AUC-PR outcome prediction validation set: {}".format(auc_pr_IB))
        print("Accuracy outcome prediction validation set: {}".format(acc_IB))
        print("F1 score outcome prediction validation set: {}".format(f1_IB))
        print("Precision outcome prediction validation set: {}".format(precision_IB))
        print("Recall outcome prediction validation set: {}".format(recall_IB))
        print("Balanced Accuracy outcome prediction validation set: {}".format(balanced_accuracy_IB))


        # Retrieving and storing dictionary of the IB metrics averaged 
        # over all test set instances
        avg_results_dict_IB = {"BCE" : avg_BCE_out_IB, 
                               "AUC-ROC" : auc_roc_IB, 
                               "AUC-PR" : auc_pr_IB, 
                               "Binary Accuracy (th 0.5)" : acc_IB, 
                               "Binary F1 (th 0.5)" : f1_IB, 
                               "Binary Precision (th 0.5)" : precision_IB, 
                               "Binary Recall (th 0.5)" : recall_IB, 
                               "Binary Bal. Accuracy (th 0.5)" : balanced_accuracy_IB}
        


    elif multic_outbool: 

        avg_CE_MCO_IB, mc_dict_IB = inf_results_IB

        acc_IB = mc_dict_IB['accuracy']

        macro_f1_IB = mc_dict_IB['macro_f1']

        weighted_f1_IB = mc_dict_IB['weighted_f1']

        macro_precision_IB = mc_dict_IB['macro_precision']

        weighted_precision_IB = mc_dict_IB['weighted_precision']

        macro_recall_IB = mc_dict_IB['macro_recall']

        weighted_recall_IB = mc_dict_IB['weighted_recall']

        print("Avg CE Multi-Class Outcome (MCO) prediction validation set: {}".format(avg_CE_MCO_IB))
        print("Accuracy MCO prediction validation set: {}".format(acc_IB))
        print("Macro F1 MCO prediction validation set: {}".format(macro_f1_IB))
        print("Weighted F1 MCO prediction validation set: {}".format(weighted_f1_IB))
        print("Macro Precision MCO prediction validation set: {}".format(macro_precision_IB))
        print("Weighted Precision MCO prediction validation set: {}".format(weighted_precision_IB))
        print("Macro Recall MCO prediction validation set: {}".format(macro_recall_IB))
        print("Weighted Recall MCO prediction validation set: {}".format(weighted_recall_IB))

        # Retrieving and storing dictionary of the IB metrics averaged 
        # over all test set instances
        avg_results_dict_IB = {"CE" : avg_CE_MCO_IB, 
                               "Multi-Class Accuracy" : acc_IB, 
                               "Macro-F1" : macro_f1_IB, 
                               "Weighted-F1" : weighted_f1_IB, 
                               "Macro-Precision" : macro_precision_IB, 
                               "Weighted-Precision" : weighted_precision_IB, 
                               "Macro-Recall" : macro_recall_IB, 
                               "Weighted-Recall" : weighted_recall_IB}
    
    path_name_average_results_IB = os.path.join(results_path, 'averaged_results_IB.pkl')

    # Writing dictionary main IB test set metrics to disk
    with open(path_name_average_results_IB, 'wb') as file:
        pickle.dump(avg_results_dict_IB, file)



    print(":::::::::::::::")
    print("CASE-BASED (CB) METRICS:")
    print(":::::::::::::::")


    # Storing average validation metrics 
    if not outcome_bool: # RRT prediction
        avg_MAE_stand_RRT_CB, avg_MAE_minutes_RRT_CB = inf_results_CB
        print("Avg MAE RRT prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_stand_RRT_CB, avg_MAE_minutes_RRT_CB))

        # Retrieving and storing dictionary of the CB metrics averaged 
        # over all test set instances
        avg_results_dict_CB = {"MAE RRT standardized" : avg_MAE_stand_RRT_CB, 
                               "MAE RRT minutes" : avg_MAE_minutes_RRT_CB}

    elif bin_outbool: 
        avg_BCE_out_CB, auc_roc_CB, auc_pr_CB, binary_dict_CB = inf_results_CB

        acc_CB = binary_dict_CB['accuracy']

        f1_CB = binary_dict_CB['f1']

        precision_CB = binary_dict_CB['precision']

        recall_CB = binary_dict_CB['recall']

        balanced_accuracy_CB  = binary_dict_CB['balanced_accuracy']

        print("Avg BCE outcome prediction validation set: {}".format(avg_BCE_out_CB))
        print("AUC-ROC outcome prediction validation set: {}".format(auc_roc_CB))
        print("AUC-PR outcome prediction validation set: {}".format(auc_pr_CB))
        print("Accuracy outcome prediction validation set: {}".format(acc_CB))
        print("F1 score outcome prediction validation set: {}".format(f1_CB))
        print("Precision outcome prediction validation set: {}".format(precision_CB))
        print("Recall outcome prediction validation set: {}".format(recall_CB))
        print("Balanced Accuracy outcome prediction validation set: {}".format(balanced_accuracy_CB))


        # Retrieving and storing dictionary of the CB metrics averaged 
        # over all test set instances
        avg_results_dict_CB = {"BCE" : avg_BCE_out_CB, 
                               "AUC-ROC" : auc_roc_CB, 
                               "AUC-PR" : auc_pr_CB, 
                               "Binary Accuracy (th 0.5)" : acc_CB, 
                               "Binary F1 (th 0.5)" : f1_CB, 
                               "Binary Precision (th 0.5)" : precision_CB, 
                               "Binary Recall (th 0.5)" : recall_CB, 
                               "Binary Bal. Accuracy (th 0.5)" : balanced_accuracy_CB}

    elif multic_outbool: 

        avg_CE_MCO_CB, mc_dict_CB = inf_results_CB

        acc_CB = mc_dict_CB['accuracy']

        macro_f1_CB = mc_dict_CB['macro_f1']

        weighted_f1_CB = mc_dict_CB['weighted_f1']

        macro_precision_CB = mc_dict_CB['macro_precision']

        weighted_precision_CB = mc_dict_CB['weighted_precision']

        macro_recall_CB = mc_dict_CB['macro_recall']

        weighted_recall_CB = mc_dict_CB['weighted_recall']

        print("Avg CE Multi-Class Outcome (MCO) prediction validation set: {}".format(avg_CE_MCO_CB))
        print("Accuracy MCO prediction validation set: {}".format(acc_CB))
        print("Macro F1 MCO prediction validation set: {}".format(macro_f1_CB))
        print("Weighted F1 MCO prediction validation set: {}".format(weighted_f1_CB))
        print("Macro Precision MCO prediction validation set: {}".format(macro_precision_CB))
        print("Weighted Precision MCO prediction validation set: {}".format(weighted_precision_CB))
        print("Macro Recall MCO prediction validation set: {}".format(macro_recall_CB))
        print("Weighted Recall MCO prediction validation set: {}".format(weighted_recall_CB))


        # Retrieving and storing dictionary of the CB metrics averaged 
        # over all test set instances
        avg_results_dict_CB = {"CE" : avg_CE_MCO_CB, 
                               "Multi-Class Accuracy" : acc_CB, 
                               "Macro-F1" : macro_f1_CB, 
                               "Weighted-F1" : weighted_f1_CB, 
                               "Macro-Precision" : macro_precision_CB, 
                               "Weighted-Precision" : weighted_precision_CB, 
                               "Macro-Recall" : macro_recall_CB, 
                               "Weighted-Recall" : weighted_recall_CB}
        

    path_name_average_results_CB = os.path.join(results_path, 'averaged_results_CB.pkl')
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
        description="Train and evaluate SuTraN using Uncertainty Weighting hyperparameter grid."
    )
    parser.add_argument("--log_name", type=str, required=True,
                        help="Dataset name (e.g., 'BPIC_17').")
    parser.add_argument("--median_caselen", type=int, required=True,
                        help="Median case length.")
    parser.add_argument("--scalar_task", type=str, required=True, 
                        help="Scalar prediction task string.")
    # parser.add_argument("--outcome_bool", type=str2bool, required=True,
    #                     help="Include outcome prediction (True/False).")
    parser.add_argument("--out_mask", type=str2bool, required=True,
                        help="Use outcome mask (True/False).")
    parser.add_argument("--final_embedding", type=str, required=True)
    parser.add_argument("--clen_dis_ref", type=str2bool, required=True,
                        help="Use CaLenDiR training (True/False).")
    # Accept num_outclasses as a string that can be an integer (e.g., '3') or 'None'
    parser.add_argument("--num_outclasses", type=str, required=True,
                        help="Number of outcome classes (e.g., '3' for multiclass) or 'None'.")
    parser.add_argument("--out_string", type=str, default="",
                        help="Optional outcome string.")
    parser.add_argument("--seed", type=int, default=24,
                        help="Seed value for reproducibility.")

    args = parser.parse_args()
    
    # Convert num_outclasses to int or None
    args.num_outclasses = str2int_or_none(args.num_outclasses)

    # Call the train_eval function with constant parameters and the parsed hyperparameters.
    train_eval(
        log_name=args.log_name,
        median_caselen=args.median_caselen,
        scalar_task=args.scalar_task,
        out_mask=args.out_mask,
        final_embedding=args.final_embedding,
        clen_dis_ref=args.clen_dis_ref,
        num_outclasses=args.num_outclasses,
        out_string=args.out_string if args.out_string != "" else None,
        seed=args.seed
    )
    


