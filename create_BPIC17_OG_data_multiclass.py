import pandas as pd 
import numpy as np
from Preprocessing.from_log_to_tensors import log_to_tensors
import os 
import torch

def preprocess_bpic17(log):
    """Preprocess the bpic17 event log. 

    Inspired by the code accompanying the following paper:

    "Irene Teinemaa, Marlon Dumas, Marcello La Rosa, and 
    Fabrizio Maria Maggi. 2019. Outcome-Oriented Predictive Process 
    Monitoring: Review and Benchmark. ACM Trans. Knowl. Discov. Data 13, 
    2, Article 17 (April 2019), 
    57 pages. https://doi.org/10.1145/3301300"

    Parameters
    ----------
    log : pandas.DataFrame 
        Event log.

    Returns
    -------
    log : pandas.DataFrame
        Preprocessed event log.
    """

    # zero based event index column
    log['event_index'] = log.groupby(['case:concept:name']).cumcount()

    # Specifying which activities (if occurring) last indicate whether a case is ...
    relevant_offer_events = ["O_Accepted", "O_Refused", "O_Cancelled"]

    # Retaining only the Offer events 
    log_offer_events = log[log['EventOrigin'] == "Offer"]

    # Getting a dataframe that gives the last Offer activity for each case, 
    # Together with the 0-based event index on which that last offer activity, 
    # determining the binary outcome, occurs. 
    last_Offer_Activities = log_offer_events.groupby('case:concept:name', sort=False).last().reset_index()[['case:concept:name','concept:name', 'event_index']]
    last_Offer_Activities.columns = ['case:concept:name', 'last_o_act', 'outc_mask_idx']

    # Adding that column as a case feature to the main log by merging on case:concept:name: 
    log = log.merge(last_Offer_Activities, on = 'case:concept:name', how = 'left')

    # Subsetting last_Offer_Activities dataframe for only the invalid cases. 
    last_Offer_Activities_invalid = last_Offer_Activities[~last_Offer_Activities['last_o_act'].isin(relevant_offer_events)]

    invalid_cases_list = list(last_Offer_Activities_invalid['case:concept:name'])

    # Dropping all invalid cases (and their events) from the main event log 
    log = log[~log['case:concept:name'].isin(invalid_cases_list)]

    # Adding the three 1-0 target columns 'case accepted', 'case refused', 'case canceled'

    # and adding another categorical case feature that contains 3 levels, indicating whether 
    # a case is 'Accepted', 'Refused' or 'Canceled':
    log['case:outcome'] = log['last_o_act'].copy()
    categorical_outcome_labels = ['Accepted', 'Refused', 'Canceled']
    binary_outcome_colnames = ['case_accepted', 'case_refused', 'case_canceled']
    for idx in range(3):
        offer_event = relevant_offer_events[idx]
        out_label = categorical_outcome_labels[idx]
        out_colname = binary_outcome_colnames[idx]
        log['case:outcome'] = np.where(log['last_o_act'] == offer_event, out_label, log['case:outcome'])
        log[out_colname] = np.where(log['last_o_act'] == offer_event, 1, 0)
    
    # Removing the two intermediate auxiliary columns 
    log = log.drop(['event_index', 'last_o_act'], axis = 1)

    # Mapping the multi-class outcome labels (case:outcome) to integer values
    out_mapping_dict = {'Accepted': 0, 'Canceled': 1, 'Refused': 2}

    log['case:outcome_int'] = log['case:outcome'].map(out_mapping_dict).astype('int32')

    return log 

def sort_log(df, case_id = 'case:concept:name', timestamp = 'time:timestamp', act_label = 'concept:name'):
    """Sort events in event log such that cases that occur first are stored 
    first, and such that events within the same case are stored based on timestamp. 

    Parameters
    ----------
    df: pd.DataFrame 
        Event log to be preprocessed. 
    case_id : str, optional
        Column name of column containing case IDs. By default 
        'case:concept:name'. 
    timestamp : str, optional
        Column name of column containing timestamps. Column Should be of 
        the datetime64 dtype. By default 'time:timestamp'. 
    act_label : str, optional
        Column name of column containing activity labels. By default 
        'concept:name'. 
    """
    df_help = df.sort_values([case_id, timestamp], ascending = [True, True], kind='mergesort').copy()
    # Now take first row of every case_id: this contains first stamp 
    df_first = df_help.drop_duplicates(subset = case_id)[[case_id, timestamp]].copy()
    df_first = df_first.sort_values(timestamp, ascending = True, kind='mergesort')
    # Include integer index to sort on. 
    df_first['case_id_int'] = [i for i in range(len(df_first))]
    df_first = df_first.drop(timestamp, axis = 1)
    df = df.merge(df_first, on = case_id, how = 'left')
    df = df.sort_values(['case_id_int', timestamp], ascending = [True, True], kind='mergesort')
    df = df.drop('case_id_int', axis = 1)
    return df.reset_index(drop=True)

def construct_BPIC17_datasets():
    temp_path = r'bpic17_with_loops.csv'
    df = pd.read_csv(temp_path, header=0)
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], format = 'mixed').dt.tz_convert('UTC')
    df = sort_log(df)
    df = preprocess_bpic17(df)

    categorical_casefeatures = ['case:LoanGoal', 'case:ApplicationType', 'lifecycle:transition']
    numeric_eventfeatures = ['FirstWithdrawalAmount', 'NumberOfTerms', 'MonthlyCost', 'OfferedAmount']
    categorical_eventfeatures = ['org:resource', 'Accepted', 'Selected']
    numeric_casefeatures = ['case:RequestedAmount']
    case_id = 'case:concept:name'
    timestamp = 'time:timestamp'
    act_label = 'concept:name'


    start_date = None
    end_date = "2017-01"
    max_days = 47.81
    window_size = 88
    log_name = 'BPIC_17'

    outcome_mask = 'outc_mask_idx'
    outcome = 'case:outcome_int'
    outcome_type = 'multiclass_outcome'


    start_before_date = None
    test_len_share = 0.25
    val_len_share = 0.2
    mode = 'preferred'
    
    train_tensors, val_tensors, test_tensors = log_to_tensors(df, 
                                                              log_name=log_name, 
                                                              start_date=start_date, 
                                                              start_before_date=start_before_date,
                                                              end_date=end_date, 
                                                              max_days=max_days, 
                                                              test_len_share=test_len_share, 
                                                              val_len_share=val_len_share, 
                                                              window_size=window_size, 
                                                              mode=mode,
                                                              case_id=case_id, 
                                                              act_label=act_label, 
                                                              timestamp=timestamp, 
                                                              cat_casefts=categorical_casefeatures, 
                                                              num_casefts=numeric_casefeatures, 
                                                              cat_eventfts=categorical_eventfeatures, 
                                                              num_eventfts=numeric_eventfeatures, 
                                                              outcome=outcome, 
                                                              outcome_type=outcome_type,
                                                              outcome_mask=outcome_mask)

    # Retrieving different elements train, validation and test set 
    train_data, og_caseint_train = train_tensors[0], train_tensors[1]
    val_data, og_caseint_val = val_tensors[0], val_tensors[1]
    test_data, og_caseint_test = test_tensors[0], test_tensors[1]


    if outcome and outcome_mask:
        instance_mask_out_train = train_tensors[-1] # shape (num_prefs, )
        instance_mask_out_val = val_tensors[-1] # shape (num_prefs, )
        instance_mask_out_test = test_tensors[-1] # shape (num_prefs, )

    # Create the log_name subfolder in the root directory of the repository
    # (Should already be created when having executed the `log_to_tensors()`
    # function.)
    output_directory = log_name
    os.makedirs(output_directory, exist_ok=True)


    # Save training tuples
    train_tensors_path = os.path.join(output_directory, 'train_tensordataset.pt')
    torch.save(train_data, train_tensors_path)

    # Save integer mapping original case id from which each instance 
    # in the train set is derived 
    og_caseint_train_path = os.path.join(output_directory, 'og_caseint_train.pt')
    torch.save(og_caseint_train, og_caseint_train_path)

    # # Save boolean tensor containing True for those instances of 
    # # which one of the prefix events already reveals the outcome 
    # outmask_path_train = os.path.join(output_directory, 'instance_mask_out_train.pt')
    # torch.save(instance_mask_out_train, outmask_path_train)



    # Save validation tuples
    val_tensors_path = os.path.join(output_directory, 'val_tensordataset.pt')
    torch.save(val_data, val_tensors_path)

    # Save integer mapping original case id from which each instance 
    # in the val set is derived 
    og_caseint_val_path = os.path.join(output_directory, 'og_caseint_val.pt')
    torch.save(og_caseint_val, og_caseint_val_path)

    # # Save boolean tensor containing True for those instances of 
    # # which one of the prefix events already reveals the outcome 
    # outmask_path_val = os.path.join(output_directory, 'instance_mask_out_val.pt')
    # torch.save(instance_mask_out_val, outmask_path_val)



    # Save test tuples
    test_tensors_path = os.path.join(output_directory, 'test_tensordataset.pt')
    torch.save(test_data, test_tensors_path)

    # Save integer mapping original case id from which each instance 
    # in the test set is derived 
    og_caseint_test_path = os.path.join(output_directory, 'og_caseint_test.pt')
    torch.save(og_caseint_test, og_caseint_test_path)

    # # Save boolean tensor containing True for those instances of 
    # # which one of the prefix events already reveals the outcome 
    # outmask_path_test = os.path.join(output_directory, 'instance_mask_out_test.pt')
    # torch.save(instance_mask_out_test, outmask_path_test)

    if outcome and outcome_mask:
        # Save boolean tensor containing True for those instances of 
        # which one of the prefix events already reveals the outcome 
        outmask_path_train = os.path.join(output_directory, 'instance_mask_out_train.pt')
        torch.save(instance_mask_out_train, outmask_path_train)

        # Save boolean tensor containing True for those instances of 
        # which one of the prefix events already reveals the outcome 
        outmask_path_val = os.path.join(output_directory, 'instance_mask_out_val.pt')
        torch.save(instance_mask_out_val, outmask_path_val)

        # Save boolean tensor containing True for those instances of 
        # which one of the prefix events already reveals the outcome 
        outmask_path_test = os.path.join(output_directory, 'instance_mask_out_test.pt')
        torch.save(instance_mask_out_test, outmask_path_test)


def main():
    construct_BPIC17_datasets()


if __name__ == '__main__':
    main()
