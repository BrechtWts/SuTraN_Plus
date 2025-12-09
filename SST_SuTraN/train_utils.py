"""Contains the loss functions for training the Sequential, 
Single-Task (SST) variants of SuTraN. This comprises the 
loss functions for:

#. Activity Suffix prediction
#. Timestamp Suffix Prediction
"""
import torch
import torch.nn as nn

from Utils.seq2seq_lossfunctions import MaskedCrossEntropyLoss, MaskedMeanAbsoluteErrorLoss
from CaLenDiR_Utils.seq2seq_norm_lossfunctions import SuffixLengthNormalizedCrossEntropyLoss, SuffixLengthNormalizedMAELoss

class SST_Loss(nn.Module): 
    def __init__(self, 
                 seq_task, 
                 clen_dis_ref, 
                 num_classes=None):
        """The all-encompassing loss function, catering to all 
        Sequential, Single-Task (SST) targets. 

        #. Activity Suffix prediction
        #. Timestamp Suffix Prediction

        Parameters
        ----------
        seq_task : {'activity_suffix', 'timestamp_suffix'}
            The (sole) sequential prediction task trained and evaluated 
            for. 
        clen_dis_ref : bool 
            If `True`, Case Length Distribution-Reflective (CaLenDiR) 
            Training is performed, and hence Suffix-Length-Normalized 
            Loss Functions are used for training. If `False`, the default 
            training procedure, in which no loss function normalization is 
            performed (and hence in which case-length distortion is not 
            addressed), is used. 
        num_classes : int or None, optional
            Number of output neurons (including padding and end tokens) 
            in the output layer of the activity suffix prediction task. 
            Only required if `seq_task='activity_suffix'`. The default of 
            None can be retained otherwise. 
        """
        super(SST_Loss, self).__init__()
        act_sufbool = (seq_task=='activity_suffix') 
        ts_sufbool = (seq_task=='timestamp_suffix')
        if not (act_sufbool or ts_sufbool):
            raise ValueError(
                "`seq_task={}` is not a valid argument. It ".format(seq_task) + 
                "should be either `'activity_suffix'` or "
                "`'timestamp_suffix'`."
            )
        if seq_task == 'activity_suffix':
            if not isinstance(num_classes, int):
                raise ValueError("When seq_task is 'activity_suffix', num_classes must be provided as an integer.")
            
        if clen_dis_ref:
            if act_sufbool: 
                self.loss_fn = SuffixLengthNormalizedCrossEntropyLoss(num_classes)
            elif ts_sufbool: 
                self.loss_fn = SuffixLengthNormalizedMAELoss()
        else:
            if act_sufbool: 
                self.loss_fn = MaskedCrossEntropyLoss(num_classes)
            elif ts_sufbool: 
                self.loss_fn = MaskedMeanAbsoluteErrorLoss()

      
    def forward(self, inputs, targets):
        """Compute the loss for the specified Sequential, 
        target for one of the two SST variants of SuTraN. 

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor containing the predictions for the respective 
            sequential target. This can be either: 
            
            #. Activity Suffix prediction of shape  
               (batch_size, window_size, num_classes). 
            
            #. Timestamp Suffix prediction of shape 
               (batch_size, window_size, 1)

        targets : torch.Tensor
            Tensor containing the labels for the respective 
            sequential target. This can be either: 

            #. Activity Suffix labels of shape  
               (batch_size, window_size) and dtype torch.int64.
            
            #. Timestamp Suffix labels of shape 
               (batch_size, window_size, 1) and dtype torch.float32.
        """
        loss = self.loss_fn(inputs, targets)
            
        return loss