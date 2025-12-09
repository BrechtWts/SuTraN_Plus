"""
GradNorm-specific loss utilities for SuTraN.

Wraps the shared multi-output loss builders and applies GradNorm
(Chen et al., 2018) to dynamically reweight the activity suffix,
timestamp suffix, remaining runtime, and optional outcome heads.

References
----------
.. [1] Chen, Z., Badrinarayanan, V., Lee, C. Y., & Rabinovich, A. (2018, July). 
       Gradnorm: Gradient normalization for adaptive loss balancing in deep 
       multitask networks. In International conference on machine learning 
       (pp. 794-803). PMLR.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from SuTraN.composite_loss_functions import MultiOutputLoss_1, MultiOutputLoss_2, MultiOutputLoss_3, MultiOutputLoss_4



###########################################################
##        Generic loss function - GradNorm wrapper       ##
###########################################################

class MultiOutputLoss_GradNorm(nn.Module):
    def __init__(self, 
                 num_classes, 
                 remaining_runtime_head, 
                 outcome_bool, 
                 clen_dis_ref, 
                 shared_params, 
                 alpha=1.5,
                 exponential_transformation=False, 
                 warmup_epoch=False, 
                 theoretical_initial_loss=False, 
                 out_mask=False, 
                 out_type=None, 
                 num_outclasses=None):
        """The all-encompassing loss function implementing GradNorm 
        weighting. 

        By default, activity and time till next event 'ttne' suffix 
        prediction tasks are always included. Additionally, the model can 
        be trained to jointly predict 1 or 2 other prediction tasks. This 
        gives us the following array of possible training setup 
        permutations.

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

        shared_params : model.shared_layer.parameters()
            Parameters of the final shared layer whose gradients feed 
            GradNorm's task-norm computation. Pass the actual parameter 
            iterator (no copies) so optimizer updates remain in sync.

        alpha : float 
            Strength restoring force GradNorm. By default 1.5. 

        exponential_transformation : bool 
            If `True`, we will learn and update the natural logarithm of 
            the weights instead of directly learning the task weights. 
            This corresponds to applying softmax normalization 
            (w = T * softmax(w)) instead of default normalization 
            (w_i = T * w_i / w.sum()) and hence initializing the 
            learnable gradnorm parameters as a 0-vector of shape 
            num_tasks, instead of a vector filled with ones. 

        warmup_epoch : bool, optional
            Boolean parameter pertaining to GradNorm optimization. 
            If `True`, a warmup epoch is utilized in which we 
            just apply equal weighting and only update the model 
            parameters based on the equally weighted composite loss. 
            The initial losses will be set to the average of the 
            individual losses over all batches processed during this 
            first warmup epoch. Utilizing a warmup epoch is optional, and 
            would deviate from the original GradNorm implementation, with 
            the aim of obtaining more robust initial losses. 
            By default `False`.

        theoretical_initial_loss : bool, optional
            Boolean parameter pertaining to GradNorm optimization. 
            If `True`, we are going to use a theoretical initial loss for 
            the categorical cross entropy loss for activity suffix 
            prediction of log(num_classes), and in case outcome 
            prediction is performed as well, also a theoretical initial 
            loss for binary or multi-class crossentropy as well. 
            By default `False`.

        out_mask : bool, optional
            If `True`, each batch includes an outcome mask tensor to suppress 
            instances whose prefixes already reveal the label from 
            contributing to the outcome loss/metrics. Defaults to `False`.

        out_type : {None, 'binary_outcome', 'multiclass_outcome'}, optional
            Type of outcome head configured for the multi-task setup.
            Only relevant when `outcome_bool=True`. Use `'binary_outcome'`
            for binary outcome prediction and `'multiclass_outcome'` for
            multi-class outcome prediction. By default `None`.

        num_outclasses : int or None, optional
            The number of outcome classes in case 
            `outcome_bool=True` and `out_type='multiclass_outcome'`. By 
            default `None`. 
        """
        super(MultiOutputLoss_GradNorm, self).__init__()
        # binary out bool
        bin_outbool = (out_type=='binary_outcome')
        # multiclass out bool
        multic_outbool = (out_type=='multiclass_outcome')
        if outcome_bool:
            self.out_mask = out_mask
            self.num_outclasses = num_outclasses
            if not (out_type == 'binary_outcome' or out_type == 'multiclass_outcome'):
                raise ValueError(
                    "When `outcome_bool=True`, `out_type` must be either "
                    "'binary_outcome' or 'multiclass_outcome'."
                )
            if multic_outbool: 
                if num_outclasses is None: 
                    raise ValueError(
                        "When `outcome_bool=True` and "
                        "`out_type='multiclass_outcome'`, "
                        "'num_outclasses' should be given an integer argument."
                    )
        else: 
            out_type = None
            self.out_mask = False
            self.num_outclasses = None

        # Creating auxiliary bools 
        self.only_rrt = (not outcome_bool) & remaining_runtime_head
        self.only_out = outcome_bool & (not remaining_runtime_head)
        self.both_not = (not outcome_bool) & (not remaining_runtime_head)
        self.both = outcome_bool & remaining_runtime_head

        self.remaining_runtime_head = remaining_runtime_head

        # Select the proper composite loss builder and record how many
        # task-specific losses GradNorm needs to reweight.

        if self.only_rrt:
            # 3 prediction tasks 
            self.num_tasks = 3
            self.composite_loss = MultiOutputLoss_1(num_classes, clen_dis_ref)

        elif self.only_out:
            # 3 prediction tasks 
            self.num_tasks = 3
            self.composite_loss = MultiOutputLoss_2(num_classes, clen_dis_ref, 
                                                    self.out_mask, out_type)
        
        elif self.both:
            # 4 prediction tasks 
            self.num_tasks = 4
            self.composite_loss = MultiOutputLoss_3(num_classes, clen_dis_ref, 
                                                    self.out_mask, out_type)

        elif self.both_not:
            # 2 prediction tasks 
            self.num_tasks = 2
            self.composite_loss = MultiOutputLoss_4(num_classes, clen_dis_ref)


        self.alpha = alpha
        # self.softmax_normalization = softmax_normalization
        self.exponential_transformation = exponential_transformation
        self.warmup_epoch = warmup_epoch
        self.theoretical_initial_loss = theoretical_initial_loss


        # Initializing the initial losses 
        self.init_losses = torch.zeros(size=(self.num_tasks,), dtype=torch.float32, device=device) # shape (num_tasks,)

        if self.theoretical_initial_loss: 
            # theoretical initial loss activity suffix prediction (cat. CE)
            num_classes_tensor = torch.tensor(data=num_classes, dtype=torch.float32, device=device)
            theor_init_loss_act = torch.log(num_classes_tensor) # scalar tensor 

            # Activity suffix prediction loss always first entry 
            self.init_losses[0] = theor_init_loss_act

            # In case outcome prediction (binary or multiclass) is enabled
            if outcome_bool:
                if bin_outbool:
                    # NOTE: should be 2 instead of 0.5. Loss is -log(1/C) = log(C), and hence this also holds for binary CE where C=2.
                    # theor_init_loss_binary_outcome = torch.log(torch.tensor(0.5, dtype=torch.float32, device=device)) # scalar tensor
                    theor_init_loss_outcome = torch.log(torch.tensor(2, dtype=torch.float32, device=device)) # scalar tensor

                else: # multi-class outcome prediction
                    # Theoretical initial loss for multi-class outcome prediction
                    theor_init_loss_outcome = torch.log(torch.tensor(num_outclasses, dtype=torch.float32, device=device)) # scalar tensor

                # Outcome prediction loss is always stored in the final entry
                self.init_losses[-1] = theor_init_loss_outcome



        if self.exponential_transformation:
            # Initialize the log weights at 0 
            self.task_weights = nn.Parameter(torch.zeros(self.num_tasks, device=device))  # Learnable task weight
        else: # default
            # Initialize the weights at 1
            self.task_weights = nn.Parameter(torch.ones(self.num_tasks, device=device))  # Learnable task weight

        self.shared_params = shared_params
        

    def set_initial_losses(self, initial_losses):
        """Set the initial losses.

        Parameters
        ----------
        initial_losses : torch.Tensor 
            Detached tensor of shape (num_tasks,), containing the 
            initial losses of the num_tasks prediction tasks. If 
            `self.warmup_epoch` evaluates to `True`, computed 
            as the averages of the individual losses reported over the 
            batches processed during the first warmup epoch, in which 
            equal weighting is performed. Otherwise, it contains the 
            initial losses reported for the very first batch in the 
            first epoch. 
        """
        # Assigning to same device as self.init_losses
        initial_losses = initial_losses.to(device)


        if self.theoretical_initial_loss: 
            # Activity suffix and (if enabled) outcome losses already use
            # their theoretical initial values set during __init__


            # Timestamp suffix prediction MAE loss always second entry
            self.init_losses[1] = initial_losses[1]

            # Remaining runtime MAE always third entry 
            if self.remaining_runtime_head:
                self.init_losses[2] = initial_losses[2]
        
        else:
            self.init_losses = initial_losses



    def forward(self, outputs, labels, initial_step, 
                instance_mask_out=None):
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
        initial_step : bool 
            The purpose of this boolean indicator parameter depends on 
            the value of `self.warmup_epoch`. 
            
            #. If `self.warmup_epoch` evaluates to `False` : If `True`, 
               it indicates that the first training iteration (batch) is 
               being processed and hence that the initial losses should 
               be computed and stored based on this first iteration only. 


            #. If `self.warmup_epoch` evaluates to `True` : If `True`, 
               it indicates that the first epoch is being processed in 
               which we just apply equal weighting and only update the 
               model parameters based on the equally weighted composite 
               loss. The initial losses will be set to the average of the 
               individual losses over all batches processed during this 
               first warmup epoch.
        instance_mask_out : {torch.Tensor, None}, optional
            Tensor of shape `(batch_size,)` and dtype `torch.bool`. 
            Contains `True` for those instances in which the outcome 
            label can directly be derived from one of the prefix events' 
            features. `False` otherwise. By default `None`. An actual 
            instance_mask_out tensor should only be passed on if both 
            `outcome_bool=True` and `out_mask=True`. Otherwise, the 
            default of `None` should be retained. 

        Returns
        -------
        tuple
            During a warmup epoch (`initial_step=True` and
            `self.warmup_epoch=True`), returns `(loss_model, loss_items)`,
            where `loss_model` is the equally weighted composite loss
            tensor and `loss_items` is the tuple of detached per-task
            losses (Python floats).
        tuple
            After warmup, returns `(loss_model, loss_grad, *loss_items,
            grad_norms, weighted_norms)` where `loss_model` and `loss_grad`
            are the tracked tensors for model and weight updates,
            `loss_items` contains the detached per-task losses, and
            `grad_norms` / `weighted_norms` are `numpy.ndarray` objects
            holding the raw and weight-adjusted gradient norms for logging.
        """

        # Retrieve the num_tasks individual losses 
        # (tuple of num_tasks scalar loss tensors)
        # losses = self.composite_loss(outputs, labels)
        if instance_mask_out != None:
            losses = self.composite_loss(outputs, labels, instance_mask_out)
        else: 
            losses = self.composite_loss(outputs, labels)

        # Stack the losses 
        task_losses = torch.stack(losses) # shape (num_tasks,)

        if initial_step: 
            # Equal weighting during whole first epoch and no weight 
            # updates if `warmup_epoch=True`
            if self.warmup_epoch:
                # Composite model loss as equally weighted sum of 
                # individual losses 
                loss_model = torch.sum(task_losses) # scalar tensor 

                # Detach individual loss components for tracking training progress 
                loss_items = tuple(loss_component.item() for loss_component in losses)
                return loss_model, loss_items
            
            # If no warmup epoch is leveraged and when processing the 
            # very first batch of the first epoch
            else: 
                initial_losses = task_losses.detach().clone()
                self.set_initial_losses(initial_losses=initial_losses)
        

        if self.exponential_transformation:
            # GradNorm adjustment: softmax normalization in case we are learning the natural 
            # logarithm of the task weights. 
            # e^(log(w)) = w, and hence this softmax normalization step boils down to 
            # computing the normalized actual weights, since 
            # softmax(log(w)) = w / w.sum()
            normed_task_weights = self.num_tasks * F.softmax(self.task_weights, dim=-1) # shape (num_tasks, )

        else: 
            # Default implementation 
            normed_task_weights = self.num_tasks * self.task_weights / torch.sum(self.task_weights, dim=-1) # shape (num_tasks,)

        # Compute normalized losses, with gradient tracking disabled (for task weight updates)
        with torch.no_grad():
            # Compute inverse training rates 
            inverse_rates = task_losses / self.init_losses # (num_tasks, )
            # Compute relative inversse training rates 
            r_i = inverse_rates / (inverse_rates.mean()) # (num_tasks, )
            # Compute coefficients in constant term gradnorm loss (r_i(t)^alpha)
            gradnorm_coefficients = torch.pow(r_i, self.alpha) # (num_tasks, )


        # Compute (unweighted) gradient norms 
        grad_norms = []
        for task_idx in range(self.num_tasks):
            # List of tensors containing gradients. If only one shared layer is 
            # given, this contains two tensors, the gradients of the weights and the 
            # gradients of the biases, having shapes (m, n) and (n,) respectively, 
            # with m being the input dimensionality of that layer and n the output 
            # dimensionality. 
            # Note that computing the norm in this manner will return a tensor detached from the 
            # current graph (which is what is needed). 
            grad_temp = list(torch.autograd.grad(task_losses[task_idx], self.shared_params, retain_graph=True))

            # Transforming it to one large 1D tensor 
            grad_temp = torch.cat([g.reshape(-1) for g in grad_temp]) # shape (m*n + n, )

            # Compute L2 norm 
            grad_norm_temp = torch.linalg.vector_norm(grad_temp) # scalar tensor 

            grad_norms.append(grad_norm_temp)
        
        # Stack the unweighted gradnorms 
        gradnorms_tensor = torch.stack(grad_norms) # shape (num_tasks,)


        # Compute weighted norms G_i (gradient tracked wrt loss weights, not wrt model parameters)
        weighted_norms = gradnorms_tensor * normed_task_weights # shape (num_tasks,)

        # Compute expected weighted gradient norm G, treated as a constant and 
        # hence detached from computational graph 
        expected_norm = torch.mean(weighted_norms.detach()) # scalar tensor, detached 

        # Compute the constant term in L_grad, also detached. 
        const_term = expected_norm * gradnorm_coefficients # shape (num_tasks,)

        # Compute L_grad (loss to update the weights)
        loss_grad = torch.sum(torch.abs(weighted_norms-const_term)) # scalar tracked tensor 

        # Compute model loss (without including the normalized task weights in the 
        # computational graph to arrive at this weighted loss)
        loss_model = torch.sum(task_losses * (normed_task_weights.detach()))

        # Detach individual loss components for tracking training progress 
        loss_items = tuple(loss_component.item() for loss_component in losses)

        return (loss_model, loss_grad) + (loss_items, ) + (gradnorms_tensor.clone().cpu().numpy(), weighted_norms.detach().clone().cpu().numpy())