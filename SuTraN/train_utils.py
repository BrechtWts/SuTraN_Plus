"""
Loss builders for the Equal Weighting SuTraN baseline.

Exposes per-head loss wrappers (activity, TTNE, remaining runtime,
outcome) and the composite `MultiOutputLoss` class consumed by
`train_procedure.py` to assemble the appropriate loss tuple for a given
head configuration and CaLenDiR setting.
"""


import torch
import torch.nn as nn

#####################################
##    Individual loss functions    ##
#####################################

from Utils.seq2seq_lossfunctions import MaskedCrossEntropyLoss, MaskedMeanAbsoluteErrorLoss, MaskedBCELoss, Masked_MCO_CrossEntropyLoss
from CaLenDiR_Utils.seq2seq_norm_lossfunctions import SuffixLengthNormalizedCrossEntropyLoss, SuffixLengthNormalizedMAELoss

class RemainingRunTimeMAELoss(nn.Module):
    def __init__(self):
        super(RemainingRunTimeMAELoss, self).__init__()
        
    def forward(self, inputs, targets):
        """Computes the Mean Absolute Error (MAE) loss for the optional 
        remaining runtime predictions. This loss only takes into account  
        the predictions and labels on the first decoding step, since 
        we only want to make one remaining runtime prediction for each 
        prefix. The predictions and labels corresponding to the remaining 
        'window_size-1' decoding steps are ignored and do not contribute 
        to the input gradient. 

        Parameters
        ----------
        inputs : torch.Tensor
            Remaining-runtime predictions of shape 
            `(batch_size, window_size, 1)` where only the first decoder 
            step is meaningful. Dtype `torch.float32`.
        targets : torch.Tensor
            Remaining-runtime labels matching the prediction shape/dtype.

        Returns
        -------
        loss: torch.Tensor
            Scalar MAE loss remaining runtime prediction, computed over 
            the first decoder step only (no masking, only discarding all 
            but the first decoding step).
        """
        # Only select the predictions and labels of first dec step
        inputs = inputs[:, 0, 0] # (batch_size, )
        targets = targets[:, 0, 0] # (batch_size, )

        absolute_errors = torch.abs(inputs-targets) # (batch_size,)

        return torch.sum(absolute_errors) / absolute_errors.shape[0] # scalar tensor 


#####################################
##    Composite loss functions     ##
#####################################

# Number 1: default + rrt prediction
# Number 2: default + outcome prediction 
# Number 3: default + rrt + outcome prediction
# Number 4: default (only activity and timestamp suffix prediction)

class MultiOutputLoss_1(nn.Module):
    def __init__(self, num_classes, clen_dis_ref):
        """Composite loss function for the following three jointly 
        learned prediction tasks: 

        #. activity suffix prediction (default)

        #. time till next event suffix prediction (default)

        #. remaining runtime prediction
        
        Parameters
        ----------
        num_classes : int
            Number of output neurons (including padding and end tokens) 
            in the output layer of the activity suffix prediction task. 
        clen_dis_ref : bool 
            If `True`, Case Length Distribution-Reflective (CaLenDiR) 
            Training is performed, and hence Suffix-Length-Normalized 
            Loss Functions are used for training. If `False`, the default 
            training procedure, in which no loss function normalization is 
            performed (and hence in which case-length distortion is not 
            addressed), is used. 
        """
        super(MultiOutputLoss_1, self).__init__()
        if clen_dis_ref:
            self.cat_loss_fn = SuffixLengthNormalizedCrossEntropyLoss(num_classes)
            self.cont_loss_fn_ttne = SuffixLengthNormalizedMAELoss()
        else:
            self.cat_loss_fn = MaskedCrossEntropyLoss(num_classes)
            self.cont_loss_fn_ttne = MaskedMeanAbsoluteErrorLoss()
        self.cont_loss_fn_rrt = RemainingRunTimeMAELoss()

    def forward(self, outputs, labels):
        """Compute composite loss (for gradient updates) and return its 
        components as python floats for tracking training progress.

        Parameters
        ----------
        outputs : tuple of torch.Tensor
            Tuple consisting of three tensors, each containing the 
            model's predictions for one of the three tasks. 
        labels : tuple of torch.Tensor
            Tuple consisting of three tensors, each containing the 
            labels for one of the three tasks.

        Returns
        -------
        loss : torch.Tensor
            Scalar tensor. Contains the composite loss that is used for 
            updating the gradients during training. Gradient tracking 
            turned on.
        cat_loss.item() : float
            Native python float. The (masked) cross entropy loss for 
            the next activity prediction head. Not used for gradient 
            updates during training, but for keeping track of the 
            different loss components during training and evaluation.
        cont_loss1.item() : float
            Native python float. The (masked) MAE loss for the time 
            till next event prediction head. Not (directly) used for 
            gradient updates during training, but for keeping track of 
            the different loss components during training and evaluation.
        cont_loss2.item() : float
            Native python float. The (masked) MAE loss for the complete 
            remaining runtime prediction head. Not (directly) used for 
            gradient updates during training, but for keeping track of 
            the different loss components during training and evaluation.
        """
        # Loss activity suffix prediction
        cat_loss = self.cat_loss_fn(outputs[0], labels[-1])
        
        # Loss Time Till Next Event (ttne) suffix prediction
        cont_loss1 = self.cont_loss_fn_ttne(outputs[1], labels[0])

        # Loss remaining runtime (rrt) prediction
        cont_loss2 = self.cont_loss_fn_rrt(outputs[2], labels[1])

        # Composite loss (used for gradient updates)
        loss = cat_loss + cont_loss1 + cont_loss2

        # Composite loss, act suffix loss, ttne loss, rrt loss
        return loss, cat_loss.item(), cont_loss1.item(), cont_loss2.item()
    

# Number 1: default + rrt prediction
# Number 2: default + outcome prediction 
# Number 3: default + rrt + outcome prediction
# Number 4: default (only activity and timestamp suffix prediction)

class MultiOutputLoss_2(nn.Module):
    def __init__(self, num_classes, clen_dis_ref, out_mask, out_type):
        """Composite loss function for the following three jointly 
        learned prediction tasks: 

        #. activity suffix prediction (default)

        #. time till next event suffix prediction (default)

        #. outcome prediction (binary or multiclass, depending on ``out_type``)
        
        Parameters
        ----------
        num_classes : int
            Number of output neurons (including padding and end tokens) 
            in the output layer of the activity suffix prediction task. 
        clen_dis_ref : bool 
            If `True`, Case Length Distribution-Reflective (CaLenDiR) 
            Training is performed, and hence Suffix-Length-Normalized 
            Loss Functions are used for training. If `False`, the default 
            training procedure, in which no loss function normalization is 
            performed (and hence in which case-length distortion is not 
            addressed), is used. 
        out_mask : bool
            Indicates whether an instance-level outcome mask is needed 
            for preventing instances, of which the inputs (prefix events) 
            contain information directly revealing the outcome label, 
            from contributing to the outcome loss 
            function (`True`), or not (`False`).
        out_type : {'binary_outcome', 'multiclass_outcome'}
            The type of outcome prediction that is being performed in the 
            multi-task setting. Only taken into account of outcome prediction 
            is included in the event log to begin with, and hence if 
            `outcome_bool=True`. If so, `'binary_outcome'` denotes binary 
            outcome (BO) prediction (binary classification), while 
            `'multiclass_outcome'` denotes multi-class outcome (MCO) 
            prediction (Multi-Class classification). 

        """
        super(MultiOutputLoss_2, self).__init__()

        self.out_mask = out_mask

        # Creating auxiliary bools 
        # binary out bool
        bin_outbool = (out_type=='binary_outcome')
        # multiclass out bool
        multic_outbool = (out_type=='multiclass_outcome')

        if clen_dis_ref:
            self.cat_loss_fn = SuffixLengthNormalizedCrossEntropyLoss(num_classes)
            self.cont_loss_fn_ttne = SuffixLengthNormalizedMAELoss()
        else:
            self.cat_loss_fn = MaskedCrossEntropyLoss(num_classes)
            self.cont_loss_fn_ttne = MaskedMeanAbsoluteErrorLoss()

        if bin_outbool:
            if self.out_mask: 
                self.out_loss_fn = MaskedBCELoss()
            else:
                self.out_loss_fn = nn.BCELoss()
        
        elif multic_outbool: 
            if self.out_mask: 
                self.out_loss_fn = Masked_MCO_CrossEntropyLoss()
                # preds of shape (B, num_outclasses), labels of shape (B, )
            else: 
                self.out_loss_fn = nn.CrossEntropyLoss()

    def forward(self, outputs, labels, instance_mask_out=None):
        """Compute composite loss (for gradient updates) and return its 
        components as python floats for tracking training progress.

        Parameters
        ----------
        outputs : tuple of torch.Tensor
            Tuple consisting of three tensors, each containing the 
            model's predictions for one of the three tasks.
        labels : tuple of torch.Tensor
            Tuple consisting of three tensors, each containing the 
            labels for one of the three tasks.
        instance_mask_out : {torch.Tensor, None}, optional
            Tensor of shape `(batch_size,)` and dtype `torch.bool`. 
            Contains `True` for those instances in which the outcome 
            label can directly be derived from one of the prefix events' 
            features. `False` otherwise. By default `None`. An actual 
            instance_mask_out tensor should only be passed on if 
            `out_mask=True`. Otherwise, the default of `None` should be 
            retained. 
        """
        # Loss activity suffix prediction
        cat_loss = self.cat_loss_fn(outputs[0], labels[-2])

        # Loss Time Till Next Event (ttne) suffix prediction
        cont_loss = self.cont_loss_fn_ttne(outputs[1], labels[0])

        # outcome prediction
        if instance_mask_out != None:
            out_loss = self.out_loss_fn(outputs[-1], labels[-1], instance_mask_out)
        
        else:
            out_loss = self.out_loss_fn(outputs[-1], labels[-1])

        # Composite loss (used for gradient updates)
        loss = cat_loss + cont_loss + out_loss

        # Composite loss, act suffix loss, ttne loss,  outcome los
        return loss, cat_loss.item(), cont_loss.item(), out_loss.item()

# Number 1: default + rrt prediction
# Number 2: default + outcome prediction 
# Number 3: default + rrt + outcome prediction
# Number 4: default (only activity and timestamp suffix prediction)

class MultiOutputLoss_3(nn.Module):
    def __init__(self, num_classes, clen_dis_ref, out_mask, out_type):
        """Composite loss function for the following four jointly learned 
        prediction tasks: 

        #. activity suffix prediction (default)

        #. time till next event suffix prediction (default)

        #. remaining runtime prediction

        #. outcome prediction (binary or multiclass, depending on ``out_type``)
        
        Parameters
        ----------
        num_classes : int
            Number of output neurons (including padding and end tokens) 
            in the output layer of the activity suffix prediction task. 
        clen_dis_ref : bool 
            If `True`, Case Length Distribution-Reflective (CaLenDiR) 
            Training is performed, and hence Suffix-Length-Normalized 
            Loss Functions are used for training. If `False`, the default 
            training procedure, in which no loss function normalization is 
            performed (and hence in which case-length distortion is not 
            addressed), is used. 
        out_mask : bool
            Indicates whether an instance-level outcome mask is needed 
            for preventing instances, of which the inputs (prefix events) 
            contain information directly revealing the outcome label, 
            from contributing to the outcome loss 
            function (`True`), or not (`False`).
        out_type : {'binary_outcome', 'multiclass_outcome'}
            The type of outcome prediction that is being performed in the 
            multi-task setting. Only taken into account of outcome prediction 
            is included in the event log to begin with, and hence if 
            `outcome_bool=True`. If so, `'binary_outcome'` denotes binary 
            outcome (BO) prediction (binary classification), while 
            `'multiclass_outcome'` denotes multi-class outcome (MCO) 
            prediction (Multi-Class classification). 
        """
        super(MultiOutputLoss_3, self).__init__()

        self.out_mask = out_mask

        # Creating auxiliary bools 
        # binary out bool
        bin_outbool = (out_type=='binary_outcome')
        # multiclass out bool
        multic_outbool = (out_type=='multiclass_outcome')
        
        if clen_dis_ref:
            self.cat_loss_fn = SuffixLengthNormalizedCrossEntropyLoss(num_classes)
            self.cont_loss_fn_ttne = SuffixLengthNormalizedMAELoss()
        else:
            self.cat_loss_fn = MaskedCrossEntropyLoss(num_classes)
            self.cont_loss_fn_ttne = MaskedMeanAbsoluteErrorLoss()

        self.cont_loss_fn_rrt = RemainingRunTimeMAELoss()

        if bin_outbool:
            if self.out_mask: 
                self.out_loss_fn = MaskedBCELoss()
            else:
                self.out_loss_fn = nn.BCELoss()
        
        elif multic_outbool: 
            if self.out_mask: 
                self.out_loss_fn = Masked_MCO_CrossEntropyLoss()
                # preds of shape (B, num_outclasses), labels of shape (B, )
            else: 
                self.out_loss_fn = nn.CrossEntropyLoss()

    def forward(self, outputs, labels, instance_mask_out=None):
        """Compute composite loss (for gradient updates) and return its 
        components as python floats for tracking training progress.

        Parameters
        ----------
        outputs : tuple of torch.Tensor
            Tuple consisting of four tensors, each containing the 
            model's predictions for one of the four tasks.
        labels : tuple of torch.Tensor
            Tuple consisting of four tensors, each containing the 
            labels for one of the four tasks.
        instance_mask_out : {torch.Tensor, None}, optional
            Tensor of shape `(batch_size,)` and dtype `torch.bool`. 
            Contains `True` for those instances in which the outcome 
            label can directly be derived from one of the prefix events' 
            features. `False` otherwise. By default `None`. An actual 
            instance_mask_out tensor should only be passed on if 
            `out_mask=True`. Otherwise, the default of `None` should be 
            retained. 
        """
        # Loss activity suffix prediction
        cat_loss = self.cat_loss_fn(outputs[0], labels[-2])

        # Loss Time Till Next Event (ttne) suffix prediction
        cont_loss1 = self.cont_loss_fn_ttne(outputs[1], labels[0])

        # Loss remaining runtime (rrt) prediction
        cont_loss2 = self.cont_loss_fn_rrt(outputs[2], labels[1])

        # outcome prediction
        if instance_mask_out != None:
            out_loss = self.out_loss_fn(outputs[-1], labels[-1], instance_mask_out)
        
        else:
            out_loss = self.out_loss_fn(outputs[-1], labels[-1])

        # Composite loss (used for gradient updates)
        loss = cat_loss + cont_loss1 + cont_loss2 + out_loss

        # Composite loss, act suffix loss, ttne loss, rrt loss, outcome los
        return loss, cat_loss.item(), cont_loss1.item(), cont_loss2.item(), out_loss.item()

# Number 1: default + rrt prediction
# Number 2: default + outcome prediction 
# Number 3: default + rrt + outcome prediction
# Number 4: default (only activity and timestamp suffix prediction)

class MultiOutputLoss_4(nn.Module):
    def __init__(self, num_classes, clen_dis_ref):
        """Composite loss function for the following two jointly 
        learned prediction tasks: 

        #. activity suffix prediction (default)

        #. time till next event suffix prediction (default)
        
        Parameters
        ----------
        num_classes : int
            Number of output neurons (including padding and end tokens) 
            in the output layer of the activity suffix prediction task. 
        clen_dis_ref : bool 
            If `True`, Case Length Distribution-Reflective (CaLenDiR) 
            Training is performed, and hence Suffix-Length-Normalized 
            Loss Functions are used for training. If `False`, the default 
            training procedure, in which no loss function normalization is 
            performed (and hence in which case-length distortion is not 
            addressed), is used. 
        """
        super(MultiOutputLoss_4, self).__init__()
        if clen_dis_ref:
            self.cat_loss_fn = SuffixLengthNormalizedCrossEntropyLoss(num_classes)
            self.cont_loss_fn_ttne = SuffixLengthNormalizedMAELoss()
        else:
            self.cat_loss_fn = MaskedCrossEntropyLoss(num_classes)
            self.cont_loss_fn_ttne = MaskedMeanAbsoluteErrorLoss()

    def forward(self, outputs, labels):
        """Compute composite loss (for gradient updates) and return its 
        components as python floats for tracking training progress.

        Parameters
        ----------
        outputs : tuple of torch.Tensor
            Tuple consisting of two tensors, each containing the 
            model's predictions for one of the two tasks. 
        labels : tuple of torch.Tensor
            Tuple consisting of two tensors, each containing the 
            labels for one of the two tasks.

        Returns
        -------
        loss : torch.Tensor
            Scalar tensor. Contains the composite loss that is used for 
            updating the gradients during training. Gradient tracking 
            turned on.
        cat_loss.item() : float
            Native python float. The (masked) cross entropy loss for 
            the next activity prediction head. Not used for gradient 
            updates during training, but for keeping track of the 
            different loss components during training and evaluation.
        cont_loss.item() : float
            Native python float. The (masked) MAE loss for the time 
            till next event prediction head. Not (directly) used for 
            gradient updates during training, but for keeping track of 
            the different loss components during training and evaluation.
        """
        # Loss activity suffix prediction
        cat_loss = self.cat_loss_fn(outputs[0], labels[-1])
        
        # Loss Time Till Next Event (ttne) suffix prediction
        cont_loss = self.cont_loss_fn_ttne(outputs[1], labels[0])

        # Composite loss (used for gradient updates)
        loss = cat_loss + cont_loss

        # Composite loss, act suffix loss, ttne loss
        return loss, cat_loss.item(), cont_loss.item()


#####################################
##      Generic loss function      ##
#####################################

class MultiOutputLoss(nn.Module):
    def __init__(self, 
                 num_classes, 
                 remaining_runtime_head, 
                 outcome_bool, 
                 clen_dis_ref, 
                 out_mask=False,
                 out_type=None):
        """The all-encompassing loss function, catering to all possible 
        permutations of jointly learned prediction tasks. 

        By default, activity and time till next event 'ttne' suffix 
        prediction tasks are always included. Additionally, the model can 
        be trained to jointly predict 1 or 2 other prediction tasks. This 
        gives us the following array of possible training setup 
        permutations

        #. rrt prediction only : `remaining_runtime_head=True` and 
           `outcome_bool=False`. Given a prefix, the model is also trained 
           to predict the total remaining runtime. 

        #. outcome prediction only : `remaining_runtime_head=False` and 
           `outcome_bool=True`. Given a prefix, the model is also trained 
           to predict the case outcome. This can be either binary outcome 
           (BO) prediction (`out_type='binary_outcome'`) or multi-class 
           outcome (MCO) prediction (`out_type='multiclass_outcome'`). 

        #. both : `remaining_runtime_head=True` and `outcome_bool=True`.
           Given a prefix, the model is also trained to predict both the 
           total remaining runtime and outcome. Outcome can be either 
           binary outcome 
           (BO) prediction (`out_type='binary_outcome'`) or multi-class 
           outcome (MCO) prediction (`out_type='multiclass_outcome'`). 

        #. no additional prediction tasks : `remaining_runtime_head=False` 
           and `outcome_bool=False`. Given a prefix, the model is only 
           trained to predict the complete suffix of activity labels and 
           timestamps (ttne). 

        Parameters
        ----------
        num_classes : int
            Number of output neurons (including padding and end tokens) 
            in the output layer of the activity suffix prediction task. 
        remaining_runtime_head : bool 
            Whether or not the model is also trained to jointly predict 
            the remaining runtime (given a prefix). 
        outcome_bool : bool 
            Whether or not the model is also trained to jointly predict 
            the case outcome (given a prefix).
        clen_dis_ref : bool 
            If `True`, Case Length Distribution-Reflective (CaLenDiR) 
            Training is performed, and hence Suffix-Length-Normalized 
            Loss Functions are used for training. If `False`, the default 
            training procedure, in which no loss function normalization is 
            performed (and hence in which case-length distortion is not 
            addressed), is used. Note that suffix length normalization 
            is only needed for the sequential prediction targets, i.e. 
            activity suffix and timestamp suffix prediction. 
        out_mask : bool, optional
            Indicates whether an instance-level outcome mask is needed 
            for preventing instances, of which the inputs (prefix events) 
            contain information directly revealing the outcome label, 
            from contributing to the outcome loss 
            function (`True`), or not (`False`). By default `False`.
        out_type : {None, 'binary_outcome', 'multiclass_outcome'}, optional
            The type of outcome prediction that is being performed in the 
            multi-task setting. Only taken into account of outcome prediction 
            is included in the event log to begin with, and hence if 
            `outcome_bool=True`. If so, `'binary_outcome'` denotes binary 
            outcome (BO) prediction (binary classification), while 
            `'multiclass_outcome'` denotes multi-class outcome (MCO) 
            prediction (Multi-Class classification). 
        """
        super(MultiOutputLoss, self).__init__()
        if outcome_bool:
            self.out_mask = out_mask
            if not (out_type == 'binary_outcome' or out_type == 'multiclass_outcome'):
                raise ValueError(
                    "When `outcome_bool=True`, `out_type` must be either "
                    "'binary_outcome' or 'multiclass_outcome'."
                )
                
        else: 
            out_type = None
            self.out_mask = False
            
        only_rrt = (not outcome_bool) & remaining_runtime_head
        only_out = outcome_bool & (not remaining_runtime_head)
        both_not = (not outcome_bool) & (not remaining_runtime_head)
        both = outcome_bool & remaining_runtime_head

        if only_rrt:
            self.composite_loss = MultiOutputLoss_1(num_classes, clen_dis_ref)

        elif only_out:
            self.composite_loss = MultiOutputLoss_2(num_classes, clen_dis_ref, 
                                                    self.out_mask, out_type=out_type)
        
        elif both:
            self.composite_loss = MultiOutputLoss_3(num_classes, clen_dis_ref, 
                                                    self.out_mask, out_type=out_type)

        elif both_not:
            self.composite_loss = MultiOutputLoss_4(num_classes, clen_dis_ref)

    def forward(self, outputs, labels, instance_mask_out=None):
        """Compute composite loss (for gradient updates) and return its 
        components as python floats for tracking training progress.

        Parameters
        ----------
        outputs : tuple of torch.Tensor
            Tuple consisting of two, three or four tensors, each 
            containing the model's predictions for one of the two / three 
            / four prediction tasks. 
        labels : tuple of torch.Tensor
            Tuple consisting of two, three or four tensors, each 
            containing the targets for one of the two / three / four 
            prediction tasks. 
        instance_mask_out : {torch.Tensor, None}, optional
            Tensor of shape `(batch_size,)` and dtype `torch.bool`. 
            Contains `True` for those instances in which the outcome 
            label can directly be derived from one of the prefix events' 
            features. `False` otherwise. By default `None`. An actual 
            instance_mask_out tensor should only be passed on if both 
            `outcome_bool=True` and `out_mask=True`. Otherwise, the 
            default of `None` should be retained. 
        """
        # First tuple component always contains the gradient tracked 
        # composite loss, the subsequent items contain the separate 
        # components used for track keeping. 
        if instance_mask_out != None:
            losses = self.composite_loss(outputs, labels, instance_mask_out)
        else: 
            losses = self.composite_loss(outputs, labels)
        return losses