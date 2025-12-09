"""Contains the loss functions for training the Non-Sequential, 
Single-Task (NSST) (Encoder-Only) variants of SuTraN. This comprises the 
loss functions for:

#. Remaining Runtime (RRT) prediction
#. Binary Outcome (BO) prediction
#. Multi-Class Outcome (MCO) prediction
"""
import torch
import torch.nn as nn

from Utils.seq2seq_lossfunctions import MaskedBCELoss, Masked_MCO_CrossEntropyLoss

class RemainingRunTimeMAELoss(nn.Module):
    def __init__(self):
        super(RemainingRunTimeMAELoss, self).__init__()
        
    def forward(self, inputs, targets):
        """Computes the Mean Absolute Error (MAE) loss for remaining 
        runtime (RRT) predictions. This loss only takes into account 
        the predictions and labels on the first decoding step, since 
        we only want to make one remaining runtime prediction for each 
        prefix. The predictions and labels corresponding to the remaining 
        'window_size-1' decoding steps are ignored and do not contribute 
        to the input gradient. 

        Parameters
        ----------
        inputs : torch.Tensor
            The tensor containing the continuous predictions for either 
            the time till next event target, or the total remaining time 
            target. Shape (batch_size, 1) 
            and dtype torch.float32.
        targets : torch.Tensor
            The continuous time prediction targets. Shape 
            (batch_size, window_size, 1), dtype torch.float32. We only 
            need the first RRT target along the central dimension. 

        Returns
        -------
        loss: torch.Tensor
            The masked MAE loss for one of the time prediction heads. 
            Scalar tensor (shape (,)) of dtype torch.float32. 
        """
        # Only select the predictions and labels of first dec step
        inputs = inputs[:, 0] # (batch_size, )
        targets = targets[:, 0, 0] # (batch_size, )

        absolute_errors = torch.abs(inputs-targets) # (batch_size,)

        return torch.sum(absolute_errors) / absolute_errors.shape[0] # scalar 

class NSST_Loss(nn.Module):
    def __init__(self, 
                 scalar_task,  
                 out_mask=False):
      """The all-encompassing loss function, catering to all 
      Non-Sequential, Single-Task (NSST) targets. 

      #. Remaining Runtime (RRT) prediction
      #. Binary Outcome (BO) prediction
      #. Multi-Class Outcome (MCO) prediction

       Parameters
       ----------
      scalar_task : {'remaining_runtime', 'binary_outcome', 'multiclass_outcome'}
         The scalar prediction task trained and evaluated for. Either 
         `'remaining_runtime'` `'binary_outcome'` or 
         `'multiclass_outcome'`.
      out_mask : bool, optional
         Indicates whether an instance-level outcome mask is needed 
         for preventing instances, of which the inputs (prefix events) 
         contain information directly revealing the outcome label, 
         from contributing to the outcome loss 
         function (`True`), or not (`False`). By default `False`.
      """
      super(NSST_Loss, self).__init__()
      outcome_bool = (scalar_task=='binary_outcome') or (scalar_task=='multiclass_outcome')

      # Only taking into account a possible out_mask if outcome 
      # prediction is requested
      if outcome_bool: 
         self.out_mask = out_mask
      else: 
         self.out_mask = False
      
      if (scalar_task=='remaining_runtime'):
         self.loss_fct = RemainingRunTimeMAELoss()
         # pred of shape (B, 1), labels (B, W, 1), shoud be cut to (B)
      elif (scalar_task=='binary_outcome'):
         if out_mask: 
            # preds of shape (B,1), labels of shape (B,1)
            self.loss_fct = MaskedBCELoss()
         else: 
            self.loss_fct = nn.BCELoss()
      elif (scalar_task=='multiclass_outcome'):
         if out_mask: 
            self.loss_fct = Masked_MCO_CrossEntropyLoss()
            # preds of shape (B, num_outclasses), labels of shape (B, )
         else:
            self.loss_fct = nn.CrossEntropyLoss()
      
    def forward(self, inputs, targets, instance_mask_out=None):
        """Compute the loss for the specified Non-Sequential, 
        targets for one of the three NSST variants of SuTraN. 

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor containing the predictions for the respective 
            non-sequential target. This can be either: 
            
            #. Scalar remaining runtime predictions of shape 
               (batch_size, 1). 
            
            #. Scalar binary outcome predictions of shape (batch_size, 1) 

            #. Scalar Multi-Class Outcome predictions of shape 
               (batch_size, num_outclasses). 

        targets : torch.Tensor
            Tensor containing the labels for the respective 
            non-sequential target. This can be either: 

            #. Scalar remaining runtime labels of shape 
               (batch_size, window_size, 1). They will still be subsetted 
               such that only the first remaining runtime label is 
               retained for each of the batch_size instances. Dtype 
               torch.float32. 
            
            #. Scalar binary outcome labels of shape (batch_size, 1) and 
               dtype torch.float32. 

            #. Scalar Multi-Class Outcome labels of shape 
               (batch_size, ). Dtype torch.int64. 

        instance_mask_out : {torch.Tensor, None}, optional
            Tensor of shape `(batch_size,)` and dtype `torch.bool`. 
            Contains `True` for those instances in which the outcome 
            label can directly be derived from one of the prefix events' 
            features. `False` otherwise. By default `None`. An actual 
            instance_mask_out tensor should only be passed on if both 
            `outcome_bool=True` and `out_mask=True`. Otherwise, the 
            default of `None` should be retained. 
        """
        if instance_mask_out != None:
            loss = self.loss_fct(inputs, targets, instance_mask_out) # scalar tensor 

        else: 
            loss = self.loss_fct(inputs, targets) # scalar tensor 
            
        return loss

   
