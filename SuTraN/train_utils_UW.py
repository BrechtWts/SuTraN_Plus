"""
Custom masked loss functions, applying the dynamic loss weighting
technique 'Uncertainty Weighting', proposed by
"Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using
uncertainty to weigh losses for scene geometry and semantics. In
Proceedings of the IEEE conference on computer vision and pattern
recognition (pp. 7482-7491)". The module also exposes the UW+ variant
introduced in the SuTraN+ paper, where uncertainty weights are
softmax-normalized via the ``softmax_normalization`` flag.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, num_classes, clen_dis_ref, softmax_normalization):
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
        softmax_normalization : bool
            When ``True``, activates the UW+ variant described in the
            SuTraN+ paper: the exp(-log(sigma)) weights are softmax-normalized
            so they sum to the number of active tasks before computing the
            composite loss. Defaults to ``False`` for the original UW
            formulation.
        """
        super(MultiOutputLoss_1, self).__init__()
        if clen_dis_ref:
            self.cat_loss_fn = SuffixLengthNormalizedCrossEntropyLoss(num_classes)
            self.cont_loss_fn_ttne = SuffixLengthNormalizedMAELoss()
        else:
            self.cat_loss_fn = MaskedCrossEntropyLoss(num_classes)
            self.cont_loss_fn_ttne = MaskedMeanAbsoluteErrorLoss()
        self.cont_loss_fn_rrt = RemainingRunTimeMAELoss()

        self.softmax_normalization = softmax_normalization

    def forward(self, outputs, labels, log_sigmas):
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
        log_sigmas : torch.nn.Parameter
            A learnable parameter tensor of shape (num_tasks,). This 
            tensor represents the log of the uncertainty parameters for 
            each task in a multi-task learning setup. Each element in 
            `log_sigmas` corresponds to a task-specific uncertainty, 
            where the value determines the weighting of that task's loss 
            during optimization. A lower value of `log_sigmas` indicates 
            higher task confidence, and vice versa. 

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

        # Stack the three losses - shape (num_tasks,) = (3, )
        losses = torch.stack([cat_loss, cont_loss1, cont_loss2]) 

        # # Exponential transformation + inverting learned log uncertainty
        # weights = torch.exp(-log_sigmas) # shape (num_tasks,)

        # if self.softmax_normalization: 
        #     # normalize the weights prior to computation composite loss 
        #     # if MultiOutputLoss_1 is used, num_tasks = 3 
        #     normed_weights = 3 * F.softmax(weights, dim=-1) # shape (num_tasks, )

        #     # Weighting losses 
        #     weighted_losses = losses * normed_weights + log_sigmas # (num_tasks, )
            
        # else: 
        #     # Weighting losses 
        #     weighted_losses = losses * weights + log_sigmas # (num_tasks, )
        if self.softmax_normalization:
            # Computing softmax normalization manually to account for corresponding 
            # transformation in the regularization terms 
            # num_tasks = 3 for MultiOutputLoss_1
            normalizer = 3 / torch.sum(torch.exp(-log_sigmas)) # scalar tensor 

            # Applying corresponding transformation on regularization terms 
            # exp(-log_sigma) + normalizer = e(-log-sigma + log(normalizer)) 
            # And hence, to arrive at softmax normalized normed_weights by simply 
            # applying torch.exp(-parameters), we should update the log_sigmas (i.e. 
            # regularization terms) as specified below
            regularization_terms = log_sigmas - torch.log(normalizer) # shape (num_tasks, )

            # Deriving softmax normalized weights based on transformed 
            # regularization terms (sums to num_tasks)
            normed_weights = torch.exp(-regularization_terms) # shape (num_tasks, )

            # Weighting losses 
            weighted_losses = losses * normed_weights + regularization_terms # shape (num_tasks, )

        else: 
            weights = torch.exp(-log_sigmas)
            weighted_losses = losses * weights + log_sigmas # shape (num_tasks, )


        # Computing total loss 
        loss = torch.sum(weighted_losses, dim=-1) # scalar tensor 

        # Composite loss, act suffix loss, ttne loss, rrt loss
        return loss, cat_loss.item(), cont_loss1.item(), cont_loss2.item()

# Number 1: default + rrt prediction
# Number 2: default + outcome prediction 
# Number 3: default + rrt + outcome prediction
# Number 4: default (only activity and timestamp suffix prediction)

class MultiOutputLoss_2(nn.Module):
    def __init__(self, 
                 num_classes, 
                 clen_dis_ref, 
                 softmax_normalization, 
                 out_mask, 
                 out_type):
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
        softmax_normalization : bool
            When ``True``, activates the UW+ variant described in the
            SuTraN+ paper: the exp(-log(sigma)) weights are softmax-normalized
            so they sum to the number of active tasks before computing the
            composite loss. Defaults to ``False`` for the original UW
            formulation.
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

        self.softmax_normalization = softmax_normalization
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

    def forward(self, outputs, labels, log_sigmas, instance_mask_out=None):
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
        log_sigmas : torch.nn.Parameter
            A learnable parameter tensor of shape (num_tasks,). This 
            tensor represents the log of the uncertainty parameters for 
            each task in a multi-task learning setup. Each element in 
            `log_sigmas` corresponds to a task-specific uncertainty, 
            where the value determines the weighting of that task's loss 
            during optimization. A lower value of `log_sigmas` indicates 
            higher task confidence, and vice versa. 
        instance_mask_out : {torch.Tensor, None}, optional
            Tensor of shape `(batch_size,)` and dtype `torch.bool`. 
            Contains `True` for those instances in which the outcome 
            label can directly be derived from one of the prefix events' 
            features. `False` otherwise. By default `None`. An actual 
            instance_mask_out tensor should only be passed on if 
            `out_mask=True`. Otherwise, the default of `None` should be 
            retained. 
        Returns
        -------
        loss : torch.Tensor
            Composite loss used for backpropagation (gradient tracking on).
        cat_loss.item() : float
            Activity-suffix loss for logging only.
        cont_loss.item() : float
            TTNE loss for logging only.
        out_loss.item() : float
            Outcome loss (binary or multiclass) for logging only.

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

        

        # Stack the three losses - shape (num_tasks,) = (3, )
        losses = torch.stack([cat_loss, cont_loss, out_loss]) 

        if self.softmax_normalization:
            # Computing softmax normalization manually to account for corresponding 
            # transformation in the regularization terms 
            # num_tasks = 3 for MultiOutputLoss_2
            normalizer = 3 / torch.sum(torch.exp(-log_sigmas)) # scalar tensor 

            # Applying corresponding transformation on regularization terms 
            # exp(-log_sigma) + normalizer = e(-log-sigma + log(normalizer)) 
            # And hence, to arrive at softmax normalized normed_weights by simply 
            # applying torch.exp(-parameters), we should update the log_sigmas (i.e. 
            # regularization terms) as specified below
            regularization_terms = log_sigmas - torch.log(normalizer) # shape (num_tasks, )

            # Deriving softmax normalized weights based on transformed 
            # regularization terms (sums to num_tasks)
            normed_weights = torch.exp(-regularization_terms) # shape (num_tasks, )

            # Weighting losses 
            weighted_losses = losses * normed_weights + regularization_terms # shape (num_tasks, )

        else: 
            weights = torch.exp(-log_sigmas)
            weighted_losses = losses * weights + log_sigmas # shape (num_tasks, )

        # Computing total loss 
        loss = torch.sum(weighted_losses, dim=-1) # scalar tensor 


        # Composite loss, act suffix loss, ttne loss, outcome los
        return loss, cat_loss.item(), cont_loss.item(), out_loss.item()

# Number 1: default + rrt prediction
# Number 2: default + outcome prediction 
# Number 3: default + rrt + outcome prediction
# Number 4: default (only activity and timestamp suffix prediction)

class MultiOutputLoss_3(nn.Module):
    def __init__(self, num_classes, clen_dis_ref, softmax_normalization, out_mask, out_type):
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
        softmax_normalization : bool
            When ``True``, activates the UW+ variant described in the
            SuTraN+ paper: the exp(-log(sigma)) weights are softmax-normalized
            so they sum to the number of active tasks before computing the
            composite loss. Defaults to ``False`` for the original UW
            formulation.
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

        self.softmax_normalization = softmax_normalization
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


    def forward(self, outputs, labels, log_sigmas, instance_mask_out=None):
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
        log_sigmas : torch.nn.Parameter
            A learnable parameter tensor of shape (num_tasks,). This 
            tensor represents the log of the uncertainty parameters for 
            each task in a multi-task learning setup. Each element in 
            `log_sigmas` corresponds to a task-specific uncertainty, 
            where the value determines the weighting of that task's loss 
            during optimization. A lower value of `log_sigmas` indicates 
            higher task confidence, and vice versa. 
        instance_mask_out : {torch.Tensor, None}, optional
            Tensor of shape `(batch_size,)` and dtype `torch.bool`. 
            Contains `True` for those instances in which the outcome 
            label can directly be derived from one of the prefix events' 
            features. `False` otherwise. By default `None`. An actual 
            instance_mask_out tensor should only be passed on if 
            `out_mask=True`. Otherwise, the default of `None` should be 
            retained. 

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
        out_loss.item() : float
            Native python float. The Binary or Multiclass Cross Entropy 
            loss for the outcome prediction head. Not (directly) used for 
            gradient updates during training, but for keeping track of 
            the different loss components during training and evaluation.
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

        # Stack the four losses - shape (num_tasks,) = (4, )
        losses = torch.stack([cat_loss, cont_loss1, cont_loss2, out_loss]) 

        if self.softmax_normalization:
            # Computing softmax normalization manually to account for corresponding 
            # transformation in the regularization terms 
            # num_tasks = 4 for MultiOutputLoss_3
            normalizer = 4 / torch.sum(torch.exp(-log_sigmas)) # scalar tensor 

            # Applying corresponding transformation on regularization terms 
            # exp(-log_sigma) + normalizer = e(-log-sigma + log(normalizer)) 
            # And hence, to arrive at softmax normalized normed_weights by simply 
            # applying torch.exp(-parameters), we should update the log_sigmas (i.e. 
            # regularization terms) as specified below
            regularization_terms = log_sigmas - torch.log(normalizer) # shape (num_tasks, )

            # Deriving softmax normalized weights based on transformed 
            # regularization terms (sums to num_tasks)
            normed_weights = torch.exp(-regularization_terms) # shape (num_tasks, )

            # Weighting losses 
            weighted_losses = losses * normed_weights + regularization_terms # shape (num_tasks, )

        else: 
            weights = torch.exp(-log_sigmas)
            weighted_losses = losses * weights + log_sigmas # shape (num_tasks, )

        # Computing total loss 
        loss = torch.sum(weighted_losses, dim=-1) # scalar tensor 

        # Composite loss, act suffix loss, ttne loss, rrt loss, outcome loss
        return loss, cat_loss.item(), cont_loss1.item(), cont_loss2.item(), out_loss.item()

# Number 1: default + rrt prediction
# Number 2: default + outcome prediction 
# Number 3: default + rrt + outcome prediction
# Number 4: default (only activity and timestamp suffix prediction)

class MultiOutputLoss_4(nn.Module):
    def __init__(self, num_classes, clen_dis_ref, softmax_normalization):
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
        softmax_normalization : bool
            When ``True``, activates the UW+ variant described in the
            SuTraN+ paper: the exp(-log(sigma)) weights are softmax-normalized
            so they sum to the number of active tasks before computing the
            composite loss. Defaults to ``False`` for the original UW
            formulation.
        """
        super(MultiOutputLoss_4, self).__init__()
        if clen_dis_ref:
            self.cat_loss_fn = SuffixLengthNormalizedCrossEntropyLoss(num_classes)
            self.cont_loss_fn_ttne = SuffixLengthNormalizedMAELoss()
        else:
            self.cat_loss_fn = MaskedCrossEntropyLoss(num_classes)
            self.cont_loss_fn_ttne = MaskedMeanAbsoluteErrorLoss()

        self.softmax_normalization = softmax_normalization


    def forward(self, outputs, labels, log_sigmas):
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
        log_sigmas : torch.nn.Parameter
            A learnable parameter tensor of shape (num_tasks,). This 
            tensor represents the log of the uncertainty parameters for 
            each task in a multi-task learning setup. Each element in 
            `log_sigmas` corresponds to a task-specific uncertainty, 
            where the value determines the weighting of that task's loss 
            during optimization. A lower value of `log_sigmas` indicates 
            higher task confidence, and vice versa. 

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

        # Stack the two losses - shape (num_tasks,) = (2, )
        losses = torch.stack([cat_loss, cont_loss]) 

        if self.softmax_normalization:
            # Computing softmax normalization manually to account for corresponding 
            # transformation in the regularization terms 
            # num_tasks = 2 for MultiOutputLoss_4
            normalizer = 2 / torch.sum(torch.exp(-log_sigmas)) # scalar tensor 

            # Applying corresponding transformation on regularization terms 
            # exp(-log_sigma) + normalizer = e(-log-sigma + log(normalizer)) 
            # And hence, to arrive at softmax normalized normed_weights by simply 
            # applying torch.exp(-parameters), we should update the log_sigmas (i.e. 
            # regularization terms) as specified below
            regularization_terms = log_sigmas - torch.log(normalizer) # shape (num_tasks, )

            # Deriving softmax normalized weights based on transformed 
            # regularization terms (sums to num_tasks)
            normed_weights = torch.exp(-regularization_terms) # shape (num_tasks, )

            # Weighting losses 
            weighted_losses = losses * normed_weights + regularization_terms # shape (num_tasks, )

        else: 
            weights = torch.exp(-log_sigmas)
            weighted_losses = losses * weights + log_sigmas # shape (num_tasks, )

        # Computing total loss 
        loss = torch.sum(weighted_losses, dim=-1) # scalar tensor 

        # Composite loss, act suffix loss, ttne loss
        return loss, cat_loss.item(), cont_loss.item()


###########################################################
##     Generic loss function - Uncertainty Weighting     ##
###########################################################

class MultiOutputLoss_UW(nn.Module):
    def __init__(self, 
                 num_classes, 
                 remaining_runtime_head, 
                 outcome_bool, 
                 clen_dis_ref, 
                 init_logsigmas, 
                 softmax_normalization=False, 
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

        init_logsigmas : float 
            Initialization value of the log uncertainty parameters. 
            Kendall et al. experimented with values ranging from 
            -2. to 5., and indicated that these task-specific weights 
            already started to converge towards the same remaining 
            evolution after only a few hundred of iterations (batches). 

        softmax_normalization : bool, optional
            When ``True``, activates the UW+ variant described in the
            SuTraN+ paper: the exp(-log(sigma)) weights are softmax-normalized
            so they sum to the number of active tasks before computing the
            composite loss. Defaults to ``False`` for the original UW
            formulation.


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
        super(MultiOutputLoss_UW, self).__init__()

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

        # Creating auxiliary bools 
        only_rrt = (not outcome_bool) & remaining_runtime_head
        only_out = outcome_bool & (not remaining_runtime_head)
        both_not = (not outcome_bool) & (not remaining_runtime_head)
        both = outcome_bool & remaining_runtime_head

        self.softmax_normalization = softmax_normalization

        # Select the appropriate multioutputloss function and initialize 
        # the learnable parameters for the uncertainties of the tasks to be 
        # learned. 

        if only_rrt:
            # 3 prediction tasks 
            self.composite_loss = MultiOutputLoss_1(num_classes, clen_dis_ref, softmax_normalization)
            self.log_sigmas = nn.Parameter(torch.tensor([init_logsigmas]*3, device=device, dtype=torch.float32))

        elif only_out:
            # 3 prediction tasks 
            self.composite_loss = MultiOutputLoss_2(num_classes, clen_dis_ref, softmax_normalization, 
                                                    self.out_mask, out_type)
            self.log_sigmas = nn.Parameter(torch.tensor([init_logsigmas]*3, device=device, dtype=torch.float32))
        
        elif both:
            # 4 prediction tasks 
            self.composite_loss = MultiOutputLoss_3(num_classes, clen_dis_ref, softmax_normalization, 
                                                    self.out_mask, out_type)
            self.log_sigmas = nn.Parameter(torch.tensor([init_logsigmas]*4, device=device, dtype=torch.float32))

        elif both_not:
            # 2 prediction tasks 
            self.composite_loss = MultiOutputLoss_4(num_classes, clen_dis_ref, softmax_normalization)
            self.log_sigmas = nn.Parameter(torch.tensor([init_logsigmas]*2, device=device, dtype=torch.float32))
        

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
        if instance_mask_out != None:
            losses = self.composite_loss(outputs, labels, self.log_sigmas, instance_mask_out)
        else:
            losses = self.composite_loss(outputs, labels, self.log_sigmas)
        return losses