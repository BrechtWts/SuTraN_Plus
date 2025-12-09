"""Module containing functionality for selecting the best epoch 
(callback), based on more than one validation metric. 
"""



import os
import pandas as pd

def select_best_epoch(df, 
                      target_metrics, 
                      epoch_col='epoch', 
                      target_weights=None, 
                      return_df=False):
    """
    Select the best epoch based on a composite normalized score computed from multiple targets.
    Each target’s performance is summarized by averaging the normalized versions of one or more
    validation metrics. The overall composite score is computed as the weighted sum of the target
    scores.

    For each metric the normalization is done as follows:
      - If the objective is 'min' (i.e. lower is better):
            normalized = (value - min_value) / (max_value - min_value)
      - If the objective is 'max' (i.e. higher is better):
            normalized = (max_value - value) / (max_value - min_value)

    When a target has more than one metric, the normalized scores are averaged so that this target’s
    influence on the overall composite score is equivalent to that of a target with a single metric.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the epoch numbers and validation metric columns.
    target_metrics : dict
        Dictionary mapping target names to a configuration dictionary. Each configuration must have
        the following keys:
        
          - "columns": list of one or more metric column names.
          - "objectives": either a single objective (str) or a list of objectives—one per column.
            Each objective must be either 'min' (if lower is better) or 'max' (if higher is better).
        
        For example, when training for three default targets and optionally a fourth:
        
        >>> target_metrics = {
        ...     "activity_suffix": {
        ...         "columns": ["Activity suffix: 1-DL (validation)"],
        ...         "objectives": "max"
        ...     },
        ...     "timestamp_suffix": {
        ...         "columns": ["TTNE - minutes MAE validation"],
        ...         "objectives": "min"
        ...     },
        ...     "remaining_runtime": {
        ...         "columns": ["RRT - mintues MAE validation"],
        ...         "objectives": "min"
        ...     },
        ...     "binary_outcome": {  # alternatively, "multi_class_outcome"
        ...         "columns": ["Binary Outcome - AUC-ROC validation", "Binary Outcome - AUC-PR validation"],
        ...         "objectives": ["max", "max"]
        ...     }
        ... }

    epoch_col : str, optional
        The name of the column in `df` that contains epoch numbers. 
        By default 'epoch'. 
    target_weights : dict, optional
        Dictionary mapping target names to weights. The composite score is computed as the sum over
        targets of:
        
            weight[target] * (average normalized score for target)
        
        If None, all targets are assigned equal weight.
    
    return_df : bool, optional
        Whether or not to return the results dataframe with the 
        computed additional columns. By default `False`. 
    
    Returns
    -------
    best_epoch : int or float
        The epoch (from `epoch_col`) with the lowest composite score.
    best_row : pandas.Series
        The row corresponding to the best epoch.
    
    Raises
    ------
    ValueError
        If any objective is not 'min' or 'max', or if a list of objectives is provided whose length
        does not match the number of metric columns for a target.
    """
    # Determine target weights. If none are provided, assign equal weight.
    if target_weights is None:
        target_weights = {target: 1.0 for target in target_metrics.keys()}
    else:
        # Ensure every target has an associated weight.
        for target in target_metrics:
            if target not in target_weights:
                target_weights[target] = 1.0

    # Work on a copy of the DataFrame to avoid modifying the original.
    df = df.copy()

    # Process each target: normalize its metric(s) and average if needed.
    for target, config in target_metrics.items():
        columns = config["columns"] # list of column names metrics for target 
        objectives = config["objectives"] # list of 'min' or 'max'

        # If a single objective (str) is provided, extend it to all columns.
        if not isinstance(objectives, list):
            objectives = [objectives] * len(columns)
        else:
            # Ensure the length of objectives matches the number of columns.
            if len(objectives) != len(columns):
                raise ValueError(f"For target '{target}', the number of objectives must match the number of columns.")

        normalized_cols = []  # To store the names of the new normalized columns.
        # Iterate over colunm names and corresponding objective for current target 
        for col, objective in zip(columns, objectives):
            col_min = df[col].min()
            col_max = df[col].max()
            # Prevent division by zero when all values are equal.
            if col_max == col_min:
                norm_values = 0.0
            else:
                if objective == "min":
                    norm_values = (df[col] - col_min) / (col_max - col_min)
                elif objective == "max":
                    norm_values = (col_max - df[col]) / (col_max - col_min)
                else:
                    raise ValueError(f"Objective for column '{col}' in target '{target}' must be either 'min' or 'max'.")
            norm_col_name = col + "_norm"
            df[norm_col_name] = norm_values
            normalized_cols.append(norm_col_name)

        # Average the normalized scores for this target.
        df[target + "_norm"] = df[normalized_cols].mean(axis=1)

    # Compute the overall composite score as the weighted sum of target scores.
    df["composite_score"] = 0.0
    for target in target_metrics.keys():
        df["composite_score"] += target_weights[target] * df[target + "_norm"]

    # Select the row (and corresponding epoch) with the lowest composite score.
    best_row = df.loc[df["composite_score"].idxmin()]
    best_epoch = best_row[epoch_col]

    if return_df: 
        return best_epoch, best_row, df
    else:
        return best_epoch, best_row
    

def get_target_metrics_dict(task_list):
    """Construct the `target_metrics` dictionary parameter needed for 
    the `select_best_epoch()` function, based on the set of 
    tasks trained and evaluated for, and specified in the `task_list` 
    list. 

    Parameters
    ----------
    task_list : list of str 
        List containing the strings of targets trained and evaluated for. 
        These are the five possible prediction targets, together with 
        their intended string representation in `task_list`. 

            #. Activity Suffix : "activity_suffix"

            #. Timestamp Suffix : "timestamp_suffix"

            #. Remaining Runtime : "remaining_runtime"

            #. Binary Outcome : "binary_outcome"

            # Multi-Class Outcome : "multiclass_outcome"

    Returns
    -------
    target_metrics : dict of dict
        Dictionary mapping target names to a configuration dictionary. 
        Each configuration must have the following keys:
        
            #.  "columns": list of one or more metric column names.

            #. "objectives": either a single objective (str) or a list of 
               objectives—one per column.Each objective must be either 
               'min' (if lower is better) or 'max' (if higher is better).
    """

    global_target_metrics = {'activity_suffix': {'columns': ['Activity suffix: 1-DL (validation)'], 
                                                 'objectives': 'max'}, 
                             'timestamp_suffix': {'columns': ['TTNE - minutes MAE validation'], 
                                                  'objectives': 'min'}, 
                             'remaining_runtime': {'columns': ['RRT - mintues MAE validation'], 
                                                   'objectives': 'min'}, 
                             'binary_outcome': {'columns': ['Binary Outcome - AUC-ROC validation', 'Binary Outcome - AUC-PR validation'], 
                                                'objectives': ['max', 'max']}, 
                             'multiclass_outcome': {'columns': ['Multi-Class Outcome - Macro-F1 score', 'Multi-Class Outcome - Weighted-F1 score'], 
                                                    'objectives': ['max', 'max']}}
    
    # Constructing final target_metrics dictionary by only 
    # selecting the task-strings - configuration dictionaries for targets 
    # specified in the `task_list` parameter. 
    target_metrics = {key: value for key, value in global_target_metrics.items() if key in task_list}

    return target_metrics
