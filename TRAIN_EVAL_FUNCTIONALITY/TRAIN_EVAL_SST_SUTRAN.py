"""Module containing entire train and evaluation pipeline for the 
Sequential Single-Task (SST) version of SuTraN, catering to the 
following tasks:
#. Activity Suffix prediction (`seq_task=='activity_suffix'`)
#. Timestamp Suffix prediction (`seq_task=='timestamp_suffix'`)
"""


import pandas as pd 
import numpy as np 
import torch 
from torch.utils.data import TensorDataset, DataLoader
import os
import pickle 


# import model 
from SST_SuTraN.SuTraN_SST import SuTraN_SST

from SST_SuTraN.train_procedure import train_model
from SST_SuTraN.inference_procedure import inference_loop
from SST_SuTraN.convert_tensordata import subset_data


from Utils.callback_selection import get_target_metrics_dict, select_best_epoch


def load_checkpoint(model, path_to_checkpoint, train_or_eval, lr):
    """Loads already trained model into memory with the 
    learned weights, as well as the optimizer in its 
    state when saving the model.

    https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html 

    Parameters
    ----------
    model : instance of the CRTP Transformer
        Should just be initialized with the correct initialization 
        arguments, like you would have done in the beginning. 
    path_to_checkpoint : string
        Exact path where the checkpoint is stored on disk. 
    train_or_eval : str, {'train', 'eval'}
        Indicating whether you want to resume training ('train') with the 
        loaded model, or you want to evaluate it ('eval'). The layers of 
        the model will be returned in the appropriate mode. 
    lr : float 
        Learning rate of the optimizer last used for training. 
    
    Returns
    -------
    model : ...
        With trained weights loaded. 
    optimizer : ... 
        With correct optimizer state loaded. 
    final_epoch_trained : int 
        Number of final epoch that the model is trained for. 
        If you want to resume training with the loaded model, 
        start from start_epoch = final_epoch_trained + 1. 
    final_loss : ... 
        Last loss of last epoch. Don't think you need it for resuming 
        training. 
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
               seq_task,
               lr=0.0002, 
               clen_dis_ref=True, 
               seed=24):
    """Entire training and evaluation pipeline for the
    Sequential Single-Task (SST) version of SuTraN, catering to the 
    following tasks:
    1. Activity Suffix prediction (`seq_task=='activity_suffix'`)
    2. Timestamp Suffix prediction (`seq_task=='timestamp_suffix'`)

    The pipeline consists of the following steps:
    1. Load the data.
    2. Initialize the model.
    3. Train the model.
    4. Evaluate the model.
    5. Save the model.

    The model is trained for maximimally 200 epochs. Early stopping is 
    triggered in case of 24 consecutive epochs without improvement in the 
    validation metric pertaining to the task (``seq_task``) at hand.

    The weights of the model are saved after each epoch, and the final 
    model version used for evaluation is chosen based on the epoch with
    the best validation metric.

    Parameters
    ----------
    log_name : str
        Name of the log file to store the training and evaluation logs.
    median_caselen : int
        Median case length of training set. 
    seq_task : {'activity_suffix', 'timestamp_suffix'}
        The (sole) sequential prediction task trained and evaluated 
        for. 
    lr : float, optional
        Learning rate of the model, by default 0.0002
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
    seed : int, optional
        Seed value to set for reproducibility. By default 24.
    """
    data_path = log_name

    storage_path = log_name

    def load_dict(path_name):
        with open(path_name, 'rb') as file:
            loaded_dict = pickle.load(file)
        
        return loaded_dict
    
    # Specifying auxiliary booleans 
    act_sufbool = (seq_task=='activity_suffix') 
    ts_sufbool = (seq_task=='timestamp_suffix')
    if not (act_sufbool or ts_sufbool):
        raise ValueError(
            "`seq_task={}` is not a valid argument. It ".format(seq_task) + 
            "should be either `'activity_suffix'` or "
            "`'timestamp_suffix'`."
        )
    
    # Reading in the data and required characteristics
    # of the data
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


    # Fixed variables 
    d_model = 32 
    num_prefix_encoder_layers = 4
    num_decoder_layers = 4
    num_heads = 8 
    layernorm_embeds = True
    dropout = 0.2
    batch_size = 128

    # specifying path results and callbacks 
    model_string = 'SuTraN_SST_'
    model_string += seq_task 
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
    
    # Initializing model
    model = SuTraN_SST(num_activities=num_activities,
                       d_model=d_model,
                       cardinality_categoricals_pref=cardinality_list_prefix,
                       num_numericals_pref=num_numericals_pref,
                       seq_task=seq_task,
                       num_prefix_encoder_layers=num_prefix_encoder_layers,
                       num_decoder_layers=num_decoder_layers,
                       num_heads=num_heads,
                       d_ff=4*d_model,
                       dropout=dropout,
                       layernorm_embeds=layernorm_embeds)
    
    # Assign to GPU 
    model.to(device)

    # Initializing optimizer and learning rate scheduler 
    decay_factor = 0.96
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_factor)

    # Subsetting the train, validation and test sets, originally 
    # constructed for the Multi-Task version of SuTraN, for the 
    # SST variant of SuTraN, given the specified prediction task 
    train_dataset = subset_data(train_dataset, num_categoricals_pref, seq_task)
    val_dataset = subset_data(val_dataset, num_categoricals_pref, seq_task)
    test_dataset = subset_data(test_dataset, num_categoricals_pref, seq_task)

    # Training procedure 
    start_epoch = 0
    num_epochs = 200 
    num_classes = num_activities 

    train_model(model, 
                optimizer=optimizer,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                start_epoch=start_epoch,
                num_epochs=num_epochs,
                seq_task=seq_task,
                num_classes=num_classes,
                path_name=backup_path,
                num_categoricals_pref=num_categoricals_pref,
                mean_std_ttne=mean_std_ttne,
                mean_std_tsp=mean_std_tsp,
                mean_std_tss=mean_std_tss,
                batch_size=batch_size,
                clen_dis_ref=clen_dis_ref,
                og_caseint_train=og_caseint_train,
                og_caseint_val=og_caseint_val,
                median_caselen=median_caselen,
                patience=24,
                lr_scheduler_present=True,
                lr_scheduler=lr_scheduler,
                seed=seed_value)
    
    # Re-initializing model for evaluation
    model = SuTraN_SST(num_activities=num_activities,
                       d_model=d_model,
                       cardinality_categoricals_pref=cardinality_list_prefix,
                       num_numericals_pref=num_numericals_pref,
                       seq_task=seq_task,
                       num_prefix_encoder_layers=num_prefix_encoder_layers,
                       num_decoder_layers=num_decoder_layers,
                       num_heads=num_heads,
                       d_ff=4*d_model,
                       dropout=dropout,
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
    target_metrics = get_target_metrics_dict([seq_task])

    # Determine number best epoch, and the corresponding row 
    # in the backup results df. 
    best_epoch, best_row = select_best_epoch(df, target_metrics)

    best_epoch = int(best_epoch)

    # The models are stored with the string underneath
    best_epoch_string = 'model_epoch_{}.pt'.format(best_epoch)
    best_epoch_path = os.path.join(backup_path, best_epoch_string)


    # Load best model into memory again 
    model, _, _, _ = load_checkpoint(model, path_to_checkpoint=best_epoch_path, train_or_eval='eval', lr=0.002)
    model.to(device)
    model.eval()

    results_path = os.path.join(backup_path, "TEST_SET_RESULTS")
    os.makedirs(results_path, exist_ok=True)

    # ------------------------
    # FINAL TEST SET INFERENCE
    # ------------------------
    
    inf_results_IB, inf_results_CB, pref_suf_results = inference_loop(model,
                                                                     test_dataset,
                                                                     seq_task=seq_task,
                                                                     num_categoricals_pref=num_categoricals_pref,
                                                                     mean_std_ttne=mean_std_ttne,
                                                                     mean_std_tsp=mean_std_tsp,
                                                                     mean_std_tss=mean_std_tss,
                                                                     og_caseint=og_caseint_test,
                                                                     results_path=results_path)
    
    print("=============================================")
    print("Final test set evaluation")
    print("=============================================")


    if act_sufbool:
        avg_dam_lev_IB = inf_results_IB[0]
        avg_results_dict_IB = {"DL sim" : avg_dam_lev_IB}

        print("IB - Avg 1-(normalized) DL distance acitivty suffix prediction validation set: {}".format(avg_dam_lev_IB))

        avg_dam_lev_CB = inf_results_CB[0]
        avg_results_dict_CB = {"DL sim" : avg_dam_lev_CB}

        print("CB - Avg 1-(normalized) DL distance acitivty suffix prediction validation set: {}".format(avg_dam_lev_CB))

    elif ts_sufbool:
        avg_MAE_ttne_stand_IB, avg_MAE_ttne_minutes_IB = inf_results_IB
        avg_results_dict_IB = {"MAE TTNE minutes" : avg_MAE_ttne_minutes_IB}
        print("IB - Avg MAE TTNE prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_ttne_stand_IB, avg_MAE_ttne_minutes_IB))

        avg_MAE_ttne_stand_CB, avg_MAE_ttne_minutes_CB = inf_results_CB
        avg_results_dict_CB = {"MAE TTNE minutes" : avg_MAE_ttne_minutes_CB}
        print("CB - Avg MAE TTNE prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_ttne_stand_CB, avg_MAE_ttne_minutes_CB))
    
    path_name_average_results_IB = os.path.join(results_path, 'averaged_results_IB.pkl')

    # Writing dictionary main IB test set metrics to disk
    with open(path_name_average_results_IB, 'wb') as file:
        pickle.dump(avg_results_dict_IB, file)
    
    path_name_average_results_CB = os.path.join(results_path, 'averaged_results_CB.pkl')
    with open(path_name_average_results_CB, 'wb') as file:
        pickle.dump(avg_results_dict_CB, file)


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

    parser = argparse.ArgumentParser(
        description="Train and evaluate SuTraN using Uncertainty Weighting hyperparameter grid."
    )
    parser.add_argument("--log_name", type=str, required=True,
                        help="Dataset name (e.g., 'BPIC_17').")
    parser.add_argument("--median_caselen", type=int, required=True,
                        help="Median case length.")
    parser.add_argument("--seq_task", type=str, required=True, 
                        help="Sequential prediction task string.")
    parser.add_argument("--clen_dis_ref", type=str2bool, required=True,
                        help="Use CaLenDiR training (True/False).")
    parser.add_argument("--seed", type=int, default=24,
                        help="Seed value for reproducibility.")

    args = parser.parse_args()

    # Call the train_eval function with constant parameters and the parsed hyperparameters.
    train_eval(
        log_name=args.log_name,
        median_caselen=args.median_caselen,
        seq_task=args.seq_task,
        clen_dis_ref=args.clen_dis_ref,
        seed=args.seed
    )





