"""
PCGrad-specific loss wrapper and gradient-projection utilities for SuTraN.

Implements Gradient Surgery (PCGrad) from Yu et al. (NeurIPS 2020) to
resolve conflicts between task-specific gradients when training the
shared Transformer backbone.

References
----------
.. [1] Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, 
        C. (2020). Gradient surgery for multi-task learning. Advances 
        in Neural Information Processing Systems, 33, 5824-5836.
"""





import torch, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from SuTraN.composite_loss_functions import MultiOutputLoss_1, MultiOutputLoss_2, MultiOutputLoss_3, MultiOutputLoss_4

###########################################################
##         Generic loss function - PCGrad wrapper        ##
###########################################################

class MultiOutputLoss_PCGrad(nn.Module):
    def __init__(self, 
                 num_classes, 
                 remaining_runtime_head, 
                 outcome_bool, 
                 clen_dis_ref, 
                 shared_layers, 
                 out_mask=False, 
                 out_type=None):
        """The all-encompassing loss function implementing PCGrad.

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
        shared_layers : 'torch.nn.modules.container.ModuleList'
            ModuleList wrapped around all the model layers shared by all 
            prediction tasks.  
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
        super(MultiOutputLoss_PCGrad, self).__init__()
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
        self.only_rrt = (not outcome_bool) & remaining_runtime_head
        self.only_out = outcome_bool & (not remaining_runtime_head)
        self.both_not = (not outcome_bool) & (not remaining_runtime_head)
        self.both = outcome_bool & remaining_runtime_head

        self.remaining_runtime_head = remaining_runtime_head


        # Select the appropriate composite loss, which determines how many
        # task heads are trained and how their individual losses are computed.
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


        self.shared_layers = shared_layers
        self.shared_params = list(shared_layers.parameters())

        # compute total dimensionality of all shared parameters 
        # combined, and a list containing the dimensionality of each 
        # shared parameter individually 
        self.grad_index, self.grad_dim, self.beg_tens, self.end_tens, self.param_size_list = self.compute_grad_dim(self.shared_params)

    def compute_grad_dim(self, shared_params):
        """Compute a number of auxiliary attributes needed during 
        training. 
        
        #. total dimensionality of the shared parameters. I.e. 
           total number of learnable parameters within the layers shared by 
           all prediction tasks. 

        Parameters
        ----------
        shared_params : list of torch.nn.parameter.Parameter
            List containing the parameter tensors contained within 
            the shared layers. 

        Returns
        -------
        grad_index : list of int  
            List containing the number of learnable parameters in each of 
            the `num_shared_params` parameter tensors pertaining to the 
            shared layers. 
        grad_dim : int 
            Total number of learnable parameters over the shared layers. 
        beg_tens : torch.Tensor 
            Tensor of dtype torch.int64 and shape `(num_shared_params,)`, 
            containing the cumulative begin index for each of the 
            respective `num_shared_params` parameter tensors, indicating 
            its first index in flattened grad tensors used during 
            each PCGrad iteration. 
        end_tens : torch.Tensor 
            Tensor of dtype torch.int64 and shape `(num_shared_params,)`, 
            containing the cumulative end index for each of the 
            respective `num_shared_params` parameter tensors, indicating 
            its last index in flattened grad tensors used during 
            each PCGrad iteration. 
        param_size_list : list of torch.Size
            List containing the original parameter tensor shapes for each 
            of the `num_shared_params` shared parameter tensors. 
        """
        # Initialize lists and tensors 
        grad_index = []
        param_size_list = []

        beg_tens = torch.zeros(size=(len(shared_params),), dtype=torch.int64, device=device) # shape (num_shared_params,)
        end_tens = torch.zeros(size=(len(shared_params),), dtype=torch.int64, device=device) # shape (num_shared_params,)

        count = 0
                
        for param in shared_params:
            grad_index.append(param.data.numel())
            param_size_list.append(param.data.size())

            if count == 0: 
                beg = 0 
            else: 
                beg = sum(grad_index[:count])
            
            end = sum(grad_index[:(count+1)])

            beg_tens[count] = beg
            end_tens[count] = end

            count += 1 

        grad_dim = sum(grad_index)

        return grad_index, grad_dim, beg_tens, end_tens, param_size_list
    
    def compute_PCGrads_trackingstats(self, 
                                      losses, 
                                      grads
                                      ):
        """Compute additional PCGrad statistics / metrics current batch 
        for further analysis. 

        Parameters
        ----------
        losses : tuple of torch.Tensor 
            Tuple of len `self.num_tasks`, containing the scalar tensors 
            of the individual task losses. 
        grads : torch.Tensor 
            Tensor of dtype torch.float32 and shape 
            `(self.num_tasks, self.grad_dim)`, containing for each of the 
            `num_tasks` the accumulated gradients of all the `grad_dim` 
            learnable parameters in the shared model layers (defined in 
            `self.shared_layers`). These are the original task-specific 
            gradients, prior to projection and hence conflict resolution. 

        Returns 
        -------
        loss_items : tuple of float 
            Tuple obtained by detaching each scalar loss tensor in the 
            `losses` tuple, and converting it to a regular python float. 
        cosine_similarity_matrix : torch.Tensor 
            Tensor of shape `(self.num_tasks, self.num_tasks)`, containing 
            a cosine similarity matrix between the task-specific gradients 
            of the shared parameters (prior to conflict resolution). E.g. 
            `cosine_similarity_matrix[i, j]` contains cosine similarity 
            between shared parameter gradients wrt task `i` vs wrt task 
            `j`, and hence between `grad[i]` and `grad[j]`. This tensor 
            will be used for incrementally updating a global tensor 
            in training procedure, and will be kept on GPU when being 
            returned (if available). 
        float_conflict_matrix : torch.Tensor 
            Tensor derived from `cosine_similarity_matrix`. Also of 
            shape `(self.num_tasks, self.num_tasks)`. Of dtype 
            torch.float32. Contains 1 on index [i,j] if gradient vector 
            wrt task i and grad vector wrt task j are conflicting (i.e. 
            negative cosine similarity), 0 otherwise. This tensor 
            will be used for incrementally updating a global tensor 
            in training procedure, and will be kept on GPU when being 
            returned (if available). 
        og_norm : numpy.ndarray
            Shape `(num_tasks,)` Contains gradient 
            magnitude (i.e. L2 norm) of each task's gradient over the
            shared layers, returned on CPU for logging.
        """
        # Detach individual loss components for tracking training progress 
        loss_items = tuple(loss_component.item() for loss_component in losses)


        # compute og gradient norms shared layers for each task loss 
        og_norm = torch.linalg.norm(grads, dim=1, keepdim=True) # shape (num_tasks, 1)

        # Cosine similarity matrix 
        #   Normalize the gradient vectors (L2 norm along the last dimension)
        grads_normalized = grads / og_norm # Shape: (num_tasks, grad_dim)

        #   Compute cosine similarity matrix
        cosine_similarity_matrix = torch.matmul(grads_normalized, grads_normalized.T)  # Shape: (num_tasks, num_tasks)

        # Derive bool conflict matrix 
        float_conflict_matrix = (cosine_similarity_matrix < 0).to(torch.float32) # shape (num_tasks, num_tasks)

        # flatten og_norm 
        og_norm = og_norm[:, 0] # shape (num_tasks,)

        return loss_items, cosine_similarity_matrix, float_conflict_matrix, og_norm.cpu().numpy()

    def forward(self, outputs, labels, instance_mask_out=None):
        """Compute composite loss (for gradient updates) and return its 
        components as python floats for tracking training progress. 
        The appropriate, potentially projected gradients, are accumulated 
        in the `grad` attribute of the model parameters. Once the method 
        has processed the batch, only the gradient update based on these 
        gradients still has to be performed (`optimizer.step()`). 

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

        Returns
        -------
        batch_weight : numpy.ndarray 
            Array of shape `(num_tasks,)`.
            A per-task scaling factor initialized to 1 for each task. It 
            accumulates adjustments based on the magnitude of conflicts 
            caused by a task's gradient when projecting other tasks' 
            gradients. Higher values indicate that a task's gradient has 
            contributed more to resolving conflicts for other tasks 
            within the batch. A value of 1 means that the task's gradient 
            did not conflict with another tasks gradient within a batch. 
        loss_items : tuple of float 
            Tuple obtained by detaching each scalar loss tensor in the 
            `losses` tuple, and converting it to a regular python float. 
        cosine_similarity_matrix : torch.Tensor 
            Tensor of shape `(self.num_tasks, self.num_tasks)`, containing 
            a cosine similarity matrix between the task-specific gradients 
            of the shared parameters (prior to conflict resolution). E.g. 
            `cosine_similarity_matrix[i, j]` contains cosine similarity 
            between shared parameter gradients wrt task `i` vs wrt task 
            `j`, and hence between `grad[i]` and `grad[j]`. This tensor 
            will be used for incrementally updating a global tensor 
            in training procedure, and will be kept on GPU when being 
            returned (if available). 
        float_conflict_matrix : torch.Tensor 
            Tensor derived from `cosine_similarity_matrix`. Also of 
            shape `(self.num_tasks, self.num_tasks)`. Of dtype 
            torch.float32. Contains 1 on index [i,j] if gradient vector 
            wrt task i and grad vector wrt task j are conflicting (i.e. 
            negative cosine similarity), 0 otherwise. This tensor 
            will be used for incrementally updating a global tensor 
            in training procedure, and will be kept on GPU when being 
            returned (if available). 
        og_norm : numpy.ndarray
            Shape `(num_tasks,)` Contains gradient 
            magnitude (i.e. L2 norm) of each task's gradient over the
            shared layers, returned on CPU for logging.
        """
        if instance_mask_out != None:
            losses = self.composite_loss(outputs, labels, instance_mask_out)
        else: 
            losses = self.composite_loss(outputs, labels)

        # Stack the losses 
        task_losses = torch.stack(losses) # shape (num_tasks,)

        # Initialize per-task batch weights and gradient accumulator
        batch_weight = np.ones(len(losses)) # shape (num_tasks,)

        # Compute gradients for each task-specific loss (i.e. initialize grads tensor prior to computation)
        grads = torch.zeros((self.num_tasks, self.grad_dim), dtype=torch.float32).to(device)  # (num_tasks, grad_dim)

        for tn in range(self.num_tasks):
            # Backpropagate loss of task tn to all layers / parameters, 
            # including prediction heads. 
            # I.e. backprop only the tn-th loss to obtain its gradient on shared layers
            task_losses[tn].backward(retain_graph=True) if (tn+1)!=self.num_tasks else task_losses[tn].backward()

            # Retrieve and flatten grads current loss for shared layer parameters 
            # only 
            grad = torch.zeros(self.grad_dim, device=device, dtype=torch.float32) # shape (grad_dim,)

            # grad2vec 
            # Retrieve, flatten and store the gradients of the shared 
            # parameters wrt the current task specific loss 
            count = 0
            for param in self.shared_params:
                grad[self.beg_tens[count]:self.end_tens[count]] = param.grad.data.view(-1)

                count += 1 
            
            # insert gradients shared parameters on index current loss 
            grads[tn] = grad

            # Reset gradients shared layers back to zero before 
            # backpropping next loss 
            self.shared_layers.zero_grad(set_to_none=False)


        # Initialize clone of the grads to start projecting 
        # and hence resolving any conflicts 
        pc_grads = grads.clone() # (num_tasks, grad_dim)


        # Loop over the num_tasks gradients and resolve conflicts 
        for tn_i in range(self.num_tasks):
            # Given the current task, iterate over all other 
            # tasks in random order 
            task_index = list(range(self.num_tasks))
            random.shuffle(task_index)
            for tn_j in task_index:
                # compute dot product 
                g_ij = torch.dot(pc_grads[tn_i], grads[tn_j]) # scalar tensor 
                # dot product of identical vectors can never be negative 
                if g_ij < 0:

                    pc_grads[tn_i] -= g_ij * grads[tn_j] / (torch.linalg.vector_norm(grads[tn_j]).pow(2) + 1e-8)

                    
                    batch_weight[tn_j] -= (g_ij / (torch.linalg.vector_norm(grads[tn_j]).pow(2) + 1e-8)).item()

        new_grads = pc_grads.sum(0)

        # Assigning accumulated projected gradients to respective 
        # shared parameters 
        count = 0 
        for param in self.shared_params: 
            beg = self.beg_tens[count]
            end = self.end_tens[count]
            param.grad.data = new_grads[beg:end].contiguous().view(self.param_size_list[count]).data.clone()


            count += 1 

        # computing additional statistics to keep track of training for further inference 
        loss_items, cosine_similarity_matrix, float_conflict_matrix, og_norm = self.compute_PCGrads_trackingstats(losses, grads)

        # Externally, you should only do a step and zero grad on the optimizer!
        return batch_weight, loss_items, cosine_similarity_matrix, float_conflict_matrix, og_norm