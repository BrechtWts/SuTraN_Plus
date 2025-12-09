"""Module containing the functionality to compute the inference metrics 
for the single task predictive models, for the scalar prediction targets 
remaining runtime (RRT), binary outcome (BO) and multi-class outcome 
(MCO) prediction. 
"""

import torch
import torch.nn as nn

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score


def compute_MCO_CE(out_pred, 
                   out_labels):
    """Compute the masked Categorical Cross-Entropy (CCE) inference 
    metric for Multi-Class Outcome (MCO) prediction. 

    Parameters
    ----------
    out_pred : torch.Tensor
        shape (batch_size, num_outclasses), unnormalized logits. 
        No softmax is applied yet. Dtype torch.float32
    out_labels : torch.Tensor
        shape (batch_size,), ground truth class indices. Dtype 
        torch.int64. 
    """
    criterion_nonred = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        loss_nonred = criterion_nonred(out_pred, out_labels) # (batch_size,)

    return loss_nonred 



def compute_outcome_BCE(out_pred, 
                        out_labels):
    """Compute the BCE inference metric for BO prediction. 

    Parameters
    ----------
    out_pred : torch.Tensor
        Tensor of shape (batch_size,) and dtype torch.float32. Containing 
        the predicted outcome probabilities for the `'batch_size'` 
        instances. 
    out_labels : torch.Tensor
        Tensor of shape (batch_size, 1) and dtype torch.float32. 
        Containing the binary outcome labels for the `'batch_size'` 
        instances. 

    Returns
    -------
    loss_nonred : torch.Tensor 
        Tensor of shape (num_prefs,), containing the BCE values for each 
        of the `'batch_size'` instances. Dtype torch.float32. 
    """
    # Also get other metrics for this - see teinemaa 
    # initializing loss computation 
    criterion_nonred = nn.BCELoss(reduction='none')
    with torch.no_grad():
        loss_nonred = criterion_nonred(out_pred.unsqueeze(-1), out_labels)[:, 0] # (batch_size,)
    
    return loss_nonred


def compute_AUC_CaseBased(out_pred, 
                          out_labels, 
                          sample_weight):
    """Compute CaLenDiR's Case-Based AUC-ROC and AUC-PR scores for 
    binary outcome (in case an outcome prediction head is included). 

    Parameters
    ----------
    out_pred : torch.Tensor
        Tensor of shape (batch_size,) and dtype torch.float32. Containing 
        the predicted outcome probabilities for the `'batch_size'` 
        instances. 
    out_labels : torch.Tensor
        Tensor of shape (batch_size, 1) and dtype torch.float32. 
        Containing the binary outcome labels for the `'batch_size'` 
        instances. 
    sample_weight : torch.Tensor 
        Tensor of dtype torch.float32 and shape (num_prefs,) in case 
        binary outcome prediction is performed without leaking 
        instances, or shape (num_prefs_out,) in case leaking 
        instances are masked and discarded for binary outcome 
        prediction. It contains for each of the (subsetted) instances 
        the weight it should be given for each original test set case 
        to contribute equally. 
    """

    # Converting the tensors to numpy arrays 
    labels_np = out_labels[:, 0].numpy() # (num_prefs,) or (num_prefs_out,)
    preds_np = out_pred.numpy() # (num_prefs,) or (num_prefs_out,)

    # Converting torch.Tensor to np.array
    sample_weight = sample_weight.numpy() # (num_prefs,) or (num_prefs_out,)

    # Computing AUC ROC 
    auc_roc = roc_auc_score(labels_np, preds_np, sample_weight=sample_weight)

    # Computing AUC PR 
    precision, recall, thresholds = precision_recall_curve(labels_np, 
                                                            preds_np, 
                                                            sample_weight=sample_weight)
    auc_pr = auc(recall, precision)

    return auc_roc, auc_pr

def compute_AUC(out_pred, 
                out_labels):
    """Computes the AUC-ROC and AUC-PR for binary outcome.

    Parameters
    ----------
    out_pred : torch.Tensor
        Tensor of shape (batch_size,) and dtype torch.float32. Containing 
        the predicted outcome probabilities for the `'batch_size'` 
        instances. 
    out_labels : torch.Tensor
        Tensor of shape (batch_size, 1) and dtype torch.float32. 
        Containing the binary outcome labels for the `'batch_size'` 
        instances. 
    """
    # Converting the tensors to numpy arrays 
    labels_np = out_labels[:, 0].numpy() # (num_prefs,)
    preds_np = out_pred.numpy() # (num_prefs,)

    # Computing AUC ROC 
    auc_roc = roc_auc_score(labels_np, preds_np)

    # Computing AUC PR 
    precision, recall, thresholds = precision_recall_curve(labels_np, preds_np)
    auc_pr = auc(recall, precision)

    return auc_roc, auc_pr


def convert_to_seconds(input_tensor, 
                       mean_std_rrt):
    """Convert a tensor containing the remaining runtime predictions or 
    labels, in the standardized scale, to the original pre-pre-processing 
    scale, which is seconds. 

    Parameters
    ----------
    input_tensor : torch.Tensor
        Tensor to be converted into the original seconds scale. Shape 
        (num_prefs,). 
    mean_std_rrt : list of float
        List containing two floats, the first being the training set mean 
        remaining runtime in seconds, and the second one the standard 
        deviation of the remaining runtime in seconds. These two quantities 
        have been used to standardize the remaining runtime labels. 
    """
    train_mean = mean_std_rrt[0]
    train_std = mean_std_rrt[1]
    
    converted_tensor = input_tensor*train_std + train_mean

    # Imposing a lower bound of 0
    converted_tensor = torch.clamp(converted_tensor, min=0) # (num_prefs,)

    return converted_tensor # Same shape as input_tensor

def compute_rrt_results(rrt_pred, 
                        rrt_labels, 
                        mean_std_rrt):
    """Compute MAE for the remaining runtime predictions, in the 
    preprocessed (standardized, ~N(0,1)) scale, as well as in the 
    original scale (seconds) by de-standardizing based on the 
    training mean and standard deviation used for standardizing the 
    RRT (Remaining RunTime) target. 

    Parameters
    ----------
    rrt_pred : torch.Tensor
        Shape (num_prefs,), standardized, dtype torch.float32. 
    rrt_labels : torch.Tensor 
       Shape (num_prefs,), standardized, dtype torch.float32. 
    mean_std_rrt : list of float
        List containing two floats, the first being the training set mean 
        remaining runtime in seconds, and the second one the standard 
        deviation of the remaining runtime in seconds. These two quantities 
        have been used to standardize the remaining runtime labels. 
    """
    # De-standardizing rrt predictions to scale in seconds based on 
    # training mean and standard deviation of the RRT labels
    rrt_preds_seconds = convert_to_seconds(input_tensor=rrt_pred.clone(), 
                                           mean_std_rrt=mean_std_rrt) # (num_prefs,)

    rrt_labels_seconds = convert_to_seconds(input_tensor=rrt_labels.clone(), 
                                            mean_std_rrt=mean_std_rrt)


    # Computing absolute errors as-is, and in seconds. 
    abs_errors = torch.abs(rrt_pred - rrt_labels) # (num_prefs,)
    abs_errors_seconds = torch.abs(rrt_preds_seconds - rrt_labels_seconds) # (num_prefs,)

    return abs_errors, abs_errors_seconds # both (num_prefs,)


def compute_multiclass_metrics(out_pred_global: torch.Tensor, 
                               out_labels: torch.Tensor):
    """Compute accuracy, macro-, and weighted-F1, Precision and Recall, 
    for multi-class outcome (MCO) prediction. 
    
    Parameters
    ----------
    out_pred_global : torch.Tensor
        shape (batch_size, num_outclasses), unnormalized logits. 
        No softmax is applied yet. Dtype torch.float32
    out_labels : torch.Tensor
        shape (batch_size,), ground truth class indices. Dtype 
        torch.int64. 
    """
    
    # 1. Get predicted class by taking argmax over logits
    pred_classes = torch.argmax(out_pred_global, dim=1) # shape (batch_size,), torch.int64
    
    # 2. Convert to numpy arrays (if needed) for sklearn
    pred_classes_np = pred_classes.cpu().numpy()
    labels_np = out_labels.cpu().numpy()
    
    # 3. Compute accuracy MCO
    acc_mco = accuracy_score(labels_np, pred_classes_np)
    
    # 4. Compute F1 scores (numpy floats)
    macro_f1_mco = f1_score(labels_np, pred_classes_np, average='macro')
    weighted_f1_mco = f1_score(labels_np, pred_classes_np, average='weighted')

    # 5. Compute macro and weighted precision and recall MCO (numpy floats)
    macro_precision_mco = precision_score(labels_np, pred_classes_np, average='macro')
    macro_recall_mco = recall_score(labels_np, pred_classes_np, average='macro')

    weighted_precision_mco = precision_score(labels_np, pred_classes_np, average='weighted')
    weighted_recall_mco = recall_score(labels_np, pred_classes_np, average='weighted')

    
    
    return {
        'accuracy': acc_mco,
        'macro_f1': macro_f1_mco,
        'weighted_f1': weighted_f1_mco, 
        'macro_precision' : macro_precision_mco, 
        'weighted_precision' : weighted_precision_mco, 
        'macro_recall' : macro_recall_mco, 
        'weighted_recall' : weighted_recall_mco
    }



def compute_multiclass_metrics_CaseBased(out_pred_global: torch.Tensor, 
                                         out_labels: torch.Tensor, 
                                         sample_weight : torch.Tensor):
    """Compute CaLenDiR's Case-Based  accuracy, macro-, and weighted-F1, 
    Precision and Recall, for multi-class outcome (MCO) prediction. 
    
    Parameters
    ----------
    out_pred_global : torch.Tensor
        shape (batch_size, num_outclasses), unnormalized logits. 
        No softmax is applied yet. Dtype torch.float32
    out_labels : torch.Tensor
        shape (batch_size,), ground truth class indices. Dtype 
        torch.int64. 
    sample_weight : torch.Tensor 
        Tensor of dtype torch.float32 and shape (batch_size,). 
        It contains for each of the (subsetted) instances 
        the weight it should be given for each original test set case 
        to contribute equally. 
    """
    sample_weight = sample_weight.numpy() # (num_prefs,) or (num_prefs_out,)

    # 1. Get predicted class by taking argmax over logits
    pred_classes = torch.argmax(out_pred_global, dim=1) # shape (batch_size,), torch.int64
    
    # 2. Convert to numpy arrays (if needed) for sklearn
    pred_classes_np = pred_classes.cpu().numpy()
    labels_np = out_labels.cpu().numpy()
    
    # 3. Compute accuracy MCO
    acc_mco = accuracy_score(labels_np, pred_classes_np, sample_weight=sample_weight)
    
    # 4. Compute F1 scores (numpy floats)
    macro_f1_mco = f1_score(labels_np, pred_classes_np, average='macro', sample_weight=sample_weight)
    weighted_f1_mco = f1_score(labels_np, pred_classes_np, average='weighted', sample_weight=sample_weight)

    # 5. Compute macro and weighted precision and recall MCO (numpy floats)
    macro_precision_mco = precision_score(labels_np, pred_classes_np, average='macro', sample_weight=sample_weight)
    macro_recall_mco = recall_score(labels_np, pred_classes_np, average='macro', sample_weight=sample_weight)

    weighted_precision_mco = precision_score(labels_np, pred_classes_np, average='weighted', sample_weight=sample_weight)
    weighted_recall_mco = recall_score(labels_np, pred_classes_np, average='weighted', sample_weight=sample_weight)

    
    
    return {
        'accuracy': acc_mco,
        'macro_f1': macro_f1_mco,
        'weighted_f1': weighted_f1_mco, 
        'macro_precision' : macro_precision_mco, 
        'weighted_precision' : weighted_precision_mco, 
        'macro_recall' : macro_recall_mco, 
        'weighted_recall' : weighted_recall_mco
    }


def compute_binary_metrics_CaseBased(out_pred_prob: torch.Tensor, 
                                     out_labels: torch.Tensor, 
                                     sample_weight: torch.Tensor,
                                     threshold: float = 0.5):
    """
    Compute accuracy, F1, precision, recall for binary outcome (BO) 
    prediction.
    
    Parameters
    ----------
    out_pred_prob : torch.Tensor
        Shape (batch_size,), contains predicted probabilities
        for the positive class (already sigmoid'ed).
        Dtype torch.float32
    out_labels : torch.Tensor
        Shape (batch_size,) or (batch_size,1), ground truth labels (0 or 1).
        Dtype torch.float32 or torch.int64.
    sample_weight : torch.Tensor 
        Tensor of dtype torch.float32 and shape (batch_size,). 
        It contains for each of the (subsetted) instances 
        the weight it should be given for each original test set case 
        to contribute equally. 
    threshold : float, optional
        Probability threshold to decide positive/negative.
    
    Returns
    -------
    dict
        Dictionary with accuracy, f1, precision, recall,
        and optionally balanced_accuracy.
    """

    sample_weight = sample_weight.numpy() # (num_prefs,) or (num_prefs_out,)
    
    # 1. Flatten labels if needed and convert to int if they're floats
    labels_flat = out_labels.view(-1).long()  # shape (T,)
    
    # 2. Convert probabilities to predicted labels using threshold
    pred_binary = (out_pred_prob >= threshold).long()  # shape (T,)
    
    # 3. Convert tensors to numpy
    labels_np = labels_flat.cpu().numpy()
    pred_np = pred_binary.cpu().numpy()
    
    # 4. Compute metrics (average='binary' is default if labels are 0/1)
    acc = accuracy_score(labels_np, pred_np, sample_weight=sample_weight)
    f1 = f1_score(labels_np, pred_np, average='binary', sample_weight=sample_weight)
    precision = precision_score(labels_np, pred_np, average='binary', sample_weight=sample_weight)
    recall = recall_score(labels_np, pred_np, average='binary', sample_weight=sample_weight)
    bal_acc = balanced_accuracy_score(labels_np, pred_np, sample_weight=sample_weight)  # optional
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'balanced_accuracy': bal_acc
    }


def compute_binary_metrics(out_pred_prob: torch.Tensor, 
                           out_labels: torch.Tensor, 
                           threshold: float = 0.5):
    """
    Compute accuracy, F1, precision, recall for binary classification.
    
    Parameters
    ----------
    out_pred_prob : torch.Tensor
        Shape (batch_size,), contains predicted probabilities
        for the positive class (already sigmoid'ed).
        Dtype torch.float32
    out_labels : torch.Tensor
        Shape (batch_size,) or (batch_size,1), ground truth labels (0 or 1).
        Dtype torch.float32 or torch.int64.
    threshold : float, optional
        Probability threshold to decide positive/negative.
    
    Returns
    -------
    dict
        Dictionary with accuracy, f1, precision, recall,
        and optionally balanced_accuracy.
    """
    
    # 1. Flatten labels if needed and convert to int if they're floats
    labels_flat = out_labels.view(-1).long()  # shape (T,)
    
    # 2. Convert probabilities to predicted labels using threshold
    pred_binary = (out_pred_prob >= threshold).long()  # shape (T,)
    
    # 3. Convert tensors to numpy
    labels_np = labels_flat.cpu().numpy()
    pred_np = pred_binary.cpu().numpy()
    
    # 4. Compute metrics (average='binary' is default if labels are 0/1)
    acc = accuracy_score(labels_np, pred_np)
    f1 = f1_score(labels_np, pred_np, average='binary')
    precision = precision_score(labels_np, pred_np, average='binary')
    recall = recall_score(labels_np, pred_np, average='binary')
    bal_acc = balanced_accuracy_score(labels_np, pred_np)  # optional
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'balanced_accuracy': bal_acc
    }