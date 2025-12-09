"""
Contains the two individual loss functions for activity suffix and time 
suffix prediction used for Case Length Distribution-Reflective (CaLenDiR) 
Training.

These loss functions are Suffix-Length Normalized Loss Functions. 
"""

import torch
import torch.nn as nn

class SuffixLengthNormalizedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes):
        super(SuffixLengthNormalizedCrossEntropyLoss, self).__init__()
        # Number of activity output neurons. Includes padding token and end_token.
        self.num_classes = num_classes
        # Padding token at index 0
        self.cross_entropy_crit = nn.CrossEntropyLoss(reduction='none', ignore_index=0)
        
    def forward(self, inputs, targets):
        """Compute the Suffix-Length Categorical Cross-Entropy Loss for 
        activity suffix prediction. The loss is normalized based on the 
        lengths of the suffixes, ensuring equal contributions from 
        sequences (suffixes) of varying lengths, mitigating a bias in the 
        batch losses towards longer cases. This is achieved by averaging 
        the Cross-Entropy values for the whole suffix per instance first, 
        masking out the padding tokens as well. 
        Thereby, an average per instance is obtained, after which the 
        final loss is computed by taking the mean over all instances.

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
        avg_normalized_loss : torch.Tensor
            The suffix length normalized masked cross entropy loss for 
            the activity suffix prediction head. Scalar tensor 
            (shape (,)) of dtype torch.float32. 
        """
        # Derive batch and window size
        batch_size = inputs.shape[0]
        window_size = inputs.shape[1]

        # Reshape to (batch_size*window_size, num_classes)
        inputs = torch.reshape(inputs, shape=(-1, self.num_classes))

        # Reshape to (batch_size*window_size, )
        targets = torch.reshape(targets, shape=(-1,))

        # Compute masked CE for all activity suffix predictions
        loss = self.cross_entropy_crit(inputs, targets) # shape (batch_size*window_size,)

        # Reshape back to (batch_size, window_size)
        loss = torch.reshape(loss, (batch_size, window_size))



        # Compute actual suffix length each instance 
        suf_len = torch.sum((torch.reshape(targets, (batch_size, window_size))!=0).to(torch.float32), dim=-1) # (batch_size,)

        # Compute average loss over each instance (suffix) - suffix length normalization
        normalized_loss = loss.sum(dim=1) / suf_len

        # Compute average loss over batch
        avg_normalized_loss = normalized_loss.mean() 

        return avg_normalized_loss # scalar tensor 
    

class SuffixLengthNormalizedMAELoss(nn.Module):
    def __init__(self):
        super(SuffixLengthNormalizedMAELoss, self).__init__()
        
    def forward(self, inputs, targets):
        """Compute the Suffix-Length Normalized Mean Absolute Error (MAE) 
        loss for timestamp suffix prediction. The loss is normalized 
        based on the lengths of the suffixes, ensuring equal 
        contributions from sequences (suffixes) of varying lengths, 
        mitigating a bias in the batch losses towards longer cases. This 
        is achieved by averaging the MAE values for the whole suffix 
        per instance first, masking out the padding tokens as well. 
        Thereby, an average per instance is obtained, after which the 
        final loss is computed by taking the mean over all instances.
        
        Masking is achieved by masking out target values of -100.0, which 
        correspond to padded event tokens.

        Parameters
        ----------
        inputs : torch.Tensor
            The tensor containing the continuous predictions for the 
            timestamp suffix target. Shape (batch_size, window_size, 1) 
            and dtype torch.float32. For the CRTP-LSTM model, this tensor 
            contains the remaining runtime suffix predictions. 
        targets : torch.Tensor
            The continuous time prediction targets. Shape 
            (batch_size, window_size, 1), dtype torch.float32. 

        Returns
        -------
        batch_mae : torch.Tensor
            The average suffix-length-normalized masked MAE loss for 
            timestamp suffix (SuTraN & ED-LSTM) or remaining runtime 
            suffix prediction (CRTP-LSTM). 
            Scalar tensor (shape (,)) of dtype torch.float32. 
        """
        # Discard trailing dimension of size 1 
        inputs = inputs[:,:,0] # (batch_size, window_size)
        targets = targets[:, :, 0] # (batch_size, window_size)

        # Create mask to ignore time targets with value -100
        mask = (targets != -100).float() # (batch_size, window_size)

        # Compute absolute errors 
        absolute_erros = torch.abs(inputs-targets) # (batch_size, window_size)

        # Replace MAE values pertaining to padding events with 0 
        masked_absolute_erros = absolute_erros * mask # (batch_size, window_size)

        # Compute average loss over each instance (suffix) - suffix length normalization
        avg_instance_mae = torch.sum(masked_absolute_erros, dim=-1) / torch.sum(mask, dim=-1) # (batch_size, )

        # Compute the average loss over the batch of instances 
        batch_mae = torch.mean(avg_instance_mae) # scalar tensor 

        return batch_mae