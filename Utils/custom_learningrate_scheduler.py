
import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


def compute_number_of_steps(og_caseint_train, 
                            batch_size, 
                            num_epochs, 
                            clen_dis_ref, 
                            median_caselen):
    """Compute the total number of training steps trained for. 

    Parameters
    ----------
    og_caseint_train : torch.Tensor 
        Tensor of dtype torch.int64 and shape 
        `(N_train,)`. Contains the integer-mapped case IDs of the 
        original training set cases from which each of the `N_train` 
        instances have been derived. Used for Uniform Case-Based Sampling 
        (UCBS) in case CaLenDiR training is adopted. 
    batch_size : int
        Batch size used for training.
    num_epochs : int
        Number of epochs trained for. 
    clen_dis_ref : bool 
        If `True`, Case Length Distribution-Reflective (CaLenDiR) 
        Training is performed. This includes the application of Uniform 
        Case-Based Sampling (UCBS) of instances each epoch, and 
        Suffix-Length-Normalized Loss Functions. If `False`, the default 
        training procedure, in which all instances are used for training 
        each epoch and in which no loss function normalization is 
        performed (and hence in which case-length distortion is not 
        addressed), is performed. 

        Determines the number of instances actually used for training 
        each epoch. 
    median_caselen : int
        Median case length original cases. 

    Returns
    -------
    _type_
        _description_
    """
    if clen_dis_ref:
        # Computing number of unique cases / integers in training set 
        number_unique_cases = len(torch.unique(og_caseint_train)) 
        # Number of instances used for training each epoch under CaLenDiR
        number_training_instances = number_unique_cases * median_caselen
    
    else:
        # No sampling performed under default training procedure
        number_training_instances = len(og_caseint_train)

    # Computing number of steps per epoch
    steps_per_epoch_float = number_training_instances / batch_size

    # Final (incomplete) batch never dropped in training
    steps_per_epoch = math.ceil(steps_per_epoch_float)

    # Total number of steps trained for
    total_steps = steps_per_epoch * num_epochs

    return total_steps

class CosineWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        super(CosineWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch + 1  # because last_epoch is initialized as -1
        if current_step < self.warmup_steps:
            lr = self.base_lr * (current_step / self.warmup_steps)
        else:
            # Cosine decay from warmup_steps to total_steps, decaying to 0.
            lr = self.base_lr * 0.5 * (1 + math.cos(
                math.pi * (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            ))
        return [lr for _ in self.optimizer.param_groups]


