"""This module contains the whole preprocessing loop. It takes the 
log as input, splits it, preprocesses it, creates prefix and suffix 
dataframes for both the train and test sets and then transforms 
these into tensors. 
"""

import os 
import numpy as np 
import pandas as pd 
from Preprocessing.dataframes_pipeline import main_dataframe_pipeline
from Preprocessing.tensor_creation import generate_tensordata_train_test


def log_to_tensors(log, 
                   log_name, 
                   start_date, 
                   start_before_date, 
                   end_date, 
                   max_days, 
                   test_len_share, 
                   val_len_share, 
                   window_size, 
                   mode,
                   case_id = 'case:concept:name', 
                   act_label = 'concept:name', 
                   timestamp = 'time:timestamp',
                   cat_casefts = [], 
                   num_casefts = [], 
                   cat_eventfts = [], 
                   num_eventfts = [], 
                   outcome=None, 
                   outcome_type=None,
                   outcome_mask=None):
    """Returns train or test data in batched datasets. 

    Parameters
    ----------
    log : pd.DataFrame 
        Event log to be preprocessed. 
    log_name : str
        Name of the event log to be processed. Will be contained 
        in file names written to disk throughout the entire preprocessing 
        procedure. 
    start_date : str or None
        "MM-YYYY" format. Only cases starting after or during the 
        specified month will be retained. If `None`, no cases will be 
        discarded based on `start_date`. 
    start_before_date : str or None
        "MM-YYYY" format. If not None, cases starting after that month 
        will be removed. 
    end_date : str or None 
        "MM-YYYY" format. Only cases ending before or during the 
        specified month will be retained. If `None`, no cases will be 
        discarded based on `end_date`. 
    max_days : _type_
        _description_
    test_len_share : float
        Fraction of last occurring cases assigned to the test set in 
        the chronological train-test split. In the SuTraN paper, this 
        was set to 0.25 for all event logs. 
    val_len_share : float 
        Percentage of cases in the training set that is assigned to the 
        validation set. The validation split is performed only after 
        having performed the train-test split, and the percentage hence 
        indicates the percentage of the training set cases after having 
        split the full event log in a training and test set. In the 
        SuTraN paper, this was set to 0.20 for all event logs. 
    
    window_size : int
        The max sequence length. Traces with case length (in number of 
        events) larger than `window_size` will be discarded. For the 
        SuTraN paper, `window_size` is set to percentile 98.5 of the 
        case length distribution, i.e. the 1.5% most extreme outliers 
        in terms of case length were discarded. 
    mode : {'preferred', 'workaround'}
        Manner in which the out-of-time train-test split is performed. 
        The 'preferred' split adheres the the approach adopted from 
        Weytjens et al, and hence, for cases containing 
        events both before and after the split point, the prefix-suffix 
        pairs for which the prefix contains at least one event after the 
        split point are assigned to the test set, while the other pairs 
        pertaining to these cases are discarded. 
        If `mode='workaround'`, prefix-suffix pairs derived from those 
        overlapping cases, and for which all prefix events occur before 
        the split point, are assigned to the training set, while other 
        pairs pertaining to these cases (i.e. pairs for which one or more 
        prefix events are recorded after the plit point) are discarded. 

        Consequently, for `mode='preferred'`, the training set will only 
        contain cases having ended before the split point, while for 
        `mode='workaround'`, the test set will only contain cases having 
        started after the split point. 
    case_id : str, optional
        Column name of column containing case IDs. By default 
        'case:concept:name'. 
    act_label : str, optional
        Column name of column containing activity labels. By default 
        'concept:name'. 
    timestamp : str, optional
        Column name of column containing timestamps. Column Should be of 
        the datetime64 dtype. By default 'time:timestamp'. 
    cat_casefts : list of str, optional
        List of column names of the categorical case features. By default 
        []. All categorical case features, if any, that should be 
        included in the final dataset tensors should be specified here. 
    num_casefts : list of str, optional
        List of column names of the numeric case features. By default 
        []. All numeric case features, if any, that should be 
        included in the final dataset tensors should be specified here. 
    cat_eventfts : list of str, optional
        List of column names of the categorical event features. By default 
        []. All categorical event features, if any, that should be 
        included in the final dataset tensors should be specified here. 
    num_eventfts : list of str, optional
        List of column names of the numeric event features. By default 
        []. All numeric event features, if any, that should be 
        included in the final dataset tensors should be specified here. 
    outcome : str, optional 
        String representing the name containing the outcome column in the
        `log` dataframe. By default `None`. The default should be retained
        if the event log does not contain an outcome label, or if no tensor 
        containing the outcome labels for each prefix-suffx pair 
        (aka instance) should be generated. 
        Note: if not `None` and if `outcome_type='binary_outcome'`, the 
        respective outcome column should be a binary integer column
        (i.e. contain either 0 or 1), and the binary value should be 
        constant for every event (i.e. dataframe row) pertaining 
        to the same case ID. If not `None` and if 
        `outcome_type='multiclass_outcome'`, the respective column should
        should also be of the integer dtype, containing the already integer 
        encoded class labels. The class labels should again be
        constant for every event (i.e. dataframe row) pertaining
        to the same case ID.
    outcome_type : {'binary_outcome', 'multiclass_outcome'}, optional
        Indicates whether the ``outcome`` column is binary (default) or
        multi-class. Ignored when ``outcome=None``. `None` by default.
    outcome_mask : str, optional 
        If an outcome column is requested and hence specified for 
        the `outcome` parameter, and if this outcome is derived from 
        specific event feature information, such as, e.g., the 
        outcome labels for the BPIC17 event logs being derived from the 
        last occurring 'O activity', the `outcome_mask` parameter should 
        be given the string representing the column name of the integer 
        column containing for each case the event index of the event 
        revealing the outcome label for that case. This is needed 
        to create an outcome mask, 
        masking those instances (i.e. prefix-suffix pairs) of which 
        the prefix (model input) already contains the event revealing the 
        outcome label and hence for preventing data leakage. 
        This parameter, if specified, is only taken into account in case 
        an outcome column is given to the `outcome` parameter. The 
        default of `None` should be retained if `outcome=None` or in case 
        the outcome is not derived from information revealed in 
        any of the events, and hence in case no outcome mask is needed. 

    Returns
    -------
    train_tensors : tuple 
        Tuple containing all tensors of the training set, which contains 
        two elements if `not (outcome and outcome_mask)`: 

        #. train_tensors[0]: tuple of torch.Tensor, containing tensors 
           pertaining to prefix events (DA), suffix events, and the 
           labels. 

        #. train_tensors[1]: Tensor of dtype torch.int64 and shape 
           `(N_train,)`. Contains the integer-mapped case IDs of the 
           original training set cases from which each of the `N_train` 
           instances have been derived. Used for Uniform Case-Based 
           Sampling (UCBS) in case CaLenDiR training is adopted. This 
           pertains to the `og_caseint_train` parameter of the 
           `train_model()` function, located in the `train_procedure.py` 
           module of the different models (`SuTraN`, `CRTP_LSTM`, 
           `LSTM_seq2seq` and `OneStepAheadBenchmarks`). 

        If  `outcome and outcome_mask` (i.e. if both are not `None`), a 
        third element is returned: 

        3. train_tensors[2]: Tensor of dtype torch.bool and shape 
           `(N_train,)`. Containing bool outcome prediction mask, 
           evaluating to True if an instance 
           (/ prefix-suffix pair / prefix) should be masked from 
           contributing to the binary CE loss or to the outcome 
           prediction evaluation metrics because of the outcome label 
           being derived from event information already contained within 
           the prefix events (that serve as inputs) to the multi-task 
           model.

    val_tensors : tuple 
        Tuple containing the same two or three elements as 
        `train_tensors`, but for the validation set instead. 
        Their description is similar to the description 
        of these `train_tensors` elements provided above. 
    test_tensors : tuple 
        Tuple containing the same two or three elements as 
        `train_tensors`. But for the test set instead. 
        Their description is similar to the description 
        of these `train_tensors` elements provided above. 
    """
    # Temporary fix, variable should be removed in all other functions as well 
    log_transformed = False
    print("Generating Dataframes...")
    train_pref_suff, val_pref_suff, test_pref_suff, cardinality_dict, num_cols_dict, cat_cols_dict, train_means_dict, train_std_dict = main_dataframe_pipeline(log, 
                                                                                                                                                            log_name, 
                                                                                                                                                            start_date, 
                                                                                                                                                            start_before_date,
                                                                                                                                                            end_date, 
                                                                                                                                                            max_days, 
                                                                                                                                                            test_len_share, 
                                                                                                                                                            val_len_share,
                                                                                                                                                            window_size, 
                                                                                                                                                            log_transformed,
                                                                                                                                                            mode,
                                                                                                                                                            case_id, 
                                                                                                                                                            act_label, 
                                                                                                                                                            timestamp, 
                                                                                                                                                            cat_casefts, 
                                                                                                                                                            num_casefts, 
                                                                                                                                                            cat_eventfts, 
                                                                                                                                                            num_eventfts, 
                                                                                                                                                            outcome, 
                                                                                                                                                            outcome_mask)
    
    print("Generating Tensors...")

    train_tensors, val_tensors, test_tensors = generate_tensordata_train_test(train_pref_suff,
                                                                              val_pref_suff, 
                                                                              test_pref_suff, 
                                                                              case_id,
                                                                              act_label, 
                                                                              cardinality_dict, 
                                                                              num_cols_dict, 
                                                                              cat_cols_dict, 
                                                                              window_size, 
                                                                              outcome,
                                                                              outcome_type, 
                                                                              outcome_mask,
                                                                              log_name)
    



    # Three additional tensors, containing the original interger case ID of each instance, 
    # for the training, validation, and test set are also returned (and written to disk)
    # return train_data, val_data, test_data, og_caseint_train, og_caseint_val, og_caseint_test

    return train_tensors, val_tensors, test_tensors