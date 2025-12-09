"""
Contains the two individual loss functions for activity suffix and time 
suffix prediction used for instance-based (default) training by all 
seq2seq models (SuTraN, ED-LSTM and CRTP-LSTM). 
"""

import torch
import torch.nn as nn


class Masked_MCO_CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(Masked_MCO_CrossEntropyLoss, self).__init__()
        self.cross_entropy_crit = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, inputs, targets, instance_mask_out):
        """Compute the masked Categorical Cross-Entropy (CCE) Loss for 
        Multi-Class Outcome (MCO) prediction, in case some of the 
        instances' (aka prefix-suffix pairs) prefix (model input) events 
        already directly reveal the outcome label to be predicted. 
        These instances should be masked in the computation 
        of the MCO CrossEntropy loss, as they induce data leakage. 

        Note: one might still opt for retaining this source of leakage 
        during training to potentially foster stronger connections 
        with other prediction tasks. For evaluation (inference) however, 
        they should always be masked. 

        Parameters
        ----------
        inputs : torch.Tensor
            The tensor containing the predicted unnormalized logits for 
            each outcome class. Shape `(batch_size, num_outclasses)` 
            and dtype `torch.float32`.
        targets : torch.Tensor
            The multi-class outcome labels, containing the indices. Shape 
            (batch_size,), dtype torch.int64. The integer fall within the 
             range `[0, num_outclasses-1]`. 
        instance_mask_out : torch.Tensor 
            Tensor of shape `(batch_size,)` and dtype `torch.bool`. 
            Contains `True` for those instances in which the outcome 
            label can directly be derived from one of the prefix events' 
            features. `False` otherwise. The instances pertaining to a 
            `True` entry are masked from contributing to the loss 
            function. 

        Returns
        -------
        loss: torch.Tensor
            The masked cross entropy loss for MCO prediction. 
            Scalar tensor (shape (,)) of dtype torch.float32. 
        """
        # Computing individual CE losses all `batch_size` instances 
        loss_nonred = self.cross_entropy_crit(inputs, targets) # shape (batch_size,)


        # Deriving numeric mask, containing a weight of 0. for the to-be 
        # masked instances, and 1. for the 'non-leaking' instances. 
        numeric_mask = (instance_mask_out==False).to(torch.float32) # (batch_size,)


        # Computing and returning the batch average CE loss over the 
        # non-masked instances solely
        return torch.sum(loss_nonred * numeric_mask) / torch.sum(numeric_mask)

class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes):
        super(MaskedCrossEntropyLoss, self).__init__()
        # Number of activity output neurons. Includes padding token and end_token.
        self.num_classes = num_classes
        # Padding token at index 0
        self.cross_entropy_crit = nn.CrossEntropyLoss(ignore_index = 0)
        
    def forward(self, inputs, targets):
        """Compute the CrossEntropyLoss of the activity suffix prediction 
        head while masking the predictions coresponding to padding events. 

        Parameters
        ----------
        inputs : torch.Tensor
            The tensor containing the unnormalized logits for each 
            activity class. Shape (batch_size, window_size, num_classes) 
            and dtype torch.float32.
        targets : torch.Tensor
            The activity labels, containing the indices. Shape 
            (batch_size, window_size), dtype torch.int64. 

        Returns
        -------
        loss: torch.Tensor
            The masked cross entropy loss for the activity prediction head. 
            Scalar tensor (shape (,)) of dtype torch.float32. 
        """
        # Reshape inputs to shape (batch_size*window_size, num_classes)
        inputs = torch.reshape(input=inputs, shape=(-1, self.num_classes))
        # Reshape targets to shape (batch_size*window_size,)
        targets = torch.reshape(input=targets, shape=(-1,))

        # Compute masked loss 
        loss = self.cross_entropy_crit(inputs, targets) # scalar tensor

        return loss
    
class MaskedMeanAbsoluteErrorLoss(nn.Module):
    def __init__(self):
        super(MaskedMeanAbsoluteErrorLoss, self).__init__()
        
    def forward(self, inputs, targets):
        """Computes the Mean Absolute Error (MAE) loss in which the 
        target values of -100.0, corresponding to padded event tokens, 
        are ignored / masked and hence do not contribute to the input 
        gradient. 

        Parameters
        ----------
        inputs : torch.Tensor
            The tensor containing the continuous predictions for the 
            timestamp suffix target. Shape (batch_size, window_size, 1) 
            and dtype torch.float32. For the CRTP-LSTM model, this tensor 
            contains the remaining runtime suffix predictions. 
        targets : torch.Tensor
            The time prediction targets. Shape 
            (batch_size, window_size, 1), dtype torch.float32. 

        Returns
        -------
        loss: torch.Tensor
            The masked MAE loss for one of the time prediction heads. 
            Scalar tensor (shape (,)) of dtype torch.float32. 
        """
        # Reshape inputs to shape (batch_size*window_size,)
        inputs = torch.reshape(input=inputs, shape=(-1,))
        # Reshape targets to shape (batch_size*window_size,)
        targets= torch.reshape(input=targets, shape=(-1,))

        # Create mask to ignore time targets with value -100
        mask = (targets != -100).float()

        absolute_errors = torch.abs(inputs-targets) # (batch_size * window_size,)

        masked_absolute_errors = absolute_errors * mask # (batch_size * window_size,)

        # count: number of non-ignored targets 
        count = torch.sum(mask)

        # Compute masked loss 
        return torch.sum(masked_absolute_errors) / count 
    
class MaskedBCELoss(nn.Module):
    def __init__(self):
        super(MaskedBCELoss, self).__init__()

        # Initializing an instance of the BCE loss
        self.bce_loss_fn = nn.BCELoss(reduction='none')
        
    def forward(self, outputs, labels, instance_mask_out):
        """Compute the masked Binary Cross-Entropy (BCE) Loss for binary 
        outcome prediction, in case some of the instances' (aka 
        prefix-suffix pairs) prefix (model input) events already 
        directly reveal the binary outcome label to be predicted. 
        These instances, while still serving as valid training 
        instances for activity suffix, timestamp suffix and 
        remaining runtime prediction, should be masked in the computation 
        of the BCE loss, as they induce data leakage. 

        Note: one might still opt for retaining this source of leakage 
        during training to potentially foster stronger connections 
        with other prediction tasks. For evaluation (inference) however, 
        they should always be masked. 

        Parameters
        ----------
        outputs : torch.Tensor
            Tensor of shape `(batch_size, 1)`, containing the model's 
            predicted probabilities for binary outcome prediction. 
            Dtype `torch.float32`. 
        labels : torch.Tensor
            Tensor of shape `(batch_size, 1)`, containing the binary 
            outcome labels (0. or 1.). Dtype `torch.float32`. 
        instance_mask_out : torch.Tensor 
            Tensor of shape `(batch_size,)` and dtype `torch.bool`. 
            Contains `True` for those instances in which the outcome 
            label can directly be derived from one of the prefix events' 
            features. `False` otherwise. 

        Returns
        -------
        torch.Tensor
            The masked BCE loss for binary outcome prediction. Scalar 
            tensor of dtype torch.float32. Computed by first computing 
            the `batch_size` individual BCE values, followed by averaging  
            over all non-masked instances only. 
        """

        # Computing individual BCE values all `batch_size` instances 
        bce_loss = self.bce_loss_fn(outputs[:, 0], labels[:, 0]) # shape (batch_size, )

        # Deriving numeric mask, containing a weight of 0. for the to-be 
        # masked instances, and 1. for the 'non-leaking' instances. 
        numeric_mask = (instance_mask_out==False).to(torch.float32) # (batch_size,)

        # Computing and returning the batch average BCE loss over the 
        # non-masked instances solely
        return torch.sum(bce_loss * numeric_mask) / torch.sum(numeric_mask)