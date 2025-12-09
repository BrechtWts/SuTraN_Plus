"""
Batch-level inference utilities for SuTraN+.

Provides `BatchInference`, a CPU-oriented evaluation helper that
consumes the aggregated predictions/labels of a full validation or test
run to compute suffix, TTNE, remaining-runtime, and outcome metrics.
(Switch `device` to CUDA and move tensors to GPU if full-sequence
metrics must be computed there.)
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score

# 'uncomment' device setup line underneath for leveraging GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# comment out line underneath for leveraging GPU 
device = torch.device("cpu")

class BatchInference():
    def __init__(self,
                 preds,
                 labels, 
                 mean_std_ttne, 
                 mean_std_tsp, 
                 mean_std_tss, 
                 mean_std_rrt, 
                 remaining_runtime_head, 
                 outcome_bool):
        """Init BatchInference object for the SuTraN+ model. Modified to 
        compute the metrics over the predictions made for the whole 
        inference (validation or test) set at once. Therefore, to ensure 
        the satisfaction of possible memory constraints and given the 
        fact that these computations should only be performed once, 
        instead of for every individual batch, all tensors and 
        computations are stored and performed on the CPU, instead of 
        the GPU. For this reason, all tensors contained within the 
        `preds` and `labels` lists specified for the initialization of 
        a `BatchInference` object, should already be stored on the CPU. 

        Contains methods for computing the different 
        evaluation metrics for activity suffix, timestamp suffix, 
        remaining runtime (optional) and outcome (optional) predictions.

        Preprocesses both predictions and labels in different scales 
        and representations, upon which the methods for computing the 
        metrics can be called. 

        Parameters
        ----------
        preds : tuple of torch.Tensor
            `(suffix_acts, ttne_suffix, [rrt], [outcome])` as produced by
            `inference_procedure.py`; all tensors must live on `device`.
        labels : tuple of torch.Tensor
            `(ttne_labels, rrt_labels?, activity_labels, [outcome_labels])`
            matching the same ordering. All tensors must live on `device`.
        mean_std_ttne : list of float
            Training mean and standard deviation used to standardize the time 
            till next event (in seconds) target. Needed for re-converting 
            ttne predictions to original scale. Mean is the first entry, 
            std the second.
        mean_std_tsp : list of float
            Training mean and standard deviation used to standardize the time 
            since previous event (in seconds) feature of the decoder suffix 
            tokens. Needed for re-converting time since previous event values 
            to original scale (seconds). Mean is the first entry, std the 2nd.
        mean_std_tss : list of float
            Training mean and standard deviation used to standardize the time 
            since start (in seconds) feature of the decoder suffix 
            tokens. Needed for re-converting time since start event values 
            to original scale (seconds). Mean is the first entry, std the 2nd.
        mean_std_rrt : list of float
            List consisting of two floats, the training mean and standard 
            deviation of the remaining runtime labels (in seconds). Needed 
            for de-standardizing remaining runtime predictions and labels, 
            such that the MAE can be expressed in seconds (and minutes). 
        remaining_runtime_head : bool
            Set to `True` when the model produced direct remaining-runtime
            predictions (i.e., the RRT head was part of training);
            controls whether RRT metrics are computed.
        outcome_bool : bool
            Indicates whether outcome predictions were produced. When
            `True`, `preds`/`labels` must include the outcome entries and
            the corresponding metrics become available.


        Notes
        -----
        Additional explanations commonly referred tensor dimensionalities: 

        * `num_prefs` : the integer number of instances, aka 
          prefix-suffix pairs, contained within the inference dataset 
          for which this `BatchInference` instance is initialized. Also 
          often referred to as `batch_size` in the comment lines 
          complementing the code. 

        * `window_size` : the maximum sequence length of both the prefix 
          event sequences, as well as the generated suffix event 
          predictions. 

        * `num_activities` : the total number of possible activity labels 
          to be predicted. This includes the padding and end token. The 
          padding token will however always be masked, such that it 
          cannot be predicted. Also referred to as `num_classes`. 

        """

        # Creating auxiliary bools 
        only_rrt = (not outcome_bool) & remaining_runtime_head
        only_out = outcome_bool & (not remaining_runtime_head)
        both_not = (not outcome_bool) & (not remaining_runtime_head)
        both = outcome_bool & remaining_runtime_head

        # Retrieving the appropriate labels and predictions 
        # NOTE: num_prefs = batch_size. The two definitions are used 
        # interchangebly in the trailing comment lines denoting the shape 
        # of the tensors. 

        #   - TTNE labels and predictions (both in standardized scale, 
        #     i.e. ~N(0,1). 
        self.ttne_labels = labels[0] # (num_prefs, window_size, 1)
        self.ttne_labels = self.ttne_labels[:,:,0] # (num_prefs, window_size)
        self.suffix_ttne_preds= preds[1] # (batch_size, window_size)

        #   - Greedily predicted activity suffixes for all validation / 
        #     test instances 
        self.suffix_acts_decoded = preds[0] # (num_prefs, window_size)

        #   - Activity suffix labels, RRT predictions and labels 
        #     (optional), outcome predictions and labels (optional)
        if only_rrt: 
            self.act_labels = labels[-1] # (num_prefs, window_size)
            self.rrt_labels = labels[1] # (num_prefs, window_size, 1)
            # Selecting the RRT labels corresponding to the first 
            # decoding step only. 
            self.rrt_labels = self.rrt_labels[:, 0, 0] # (num_prefs,)

            # Direct RRT predictions (still in standardized scale)
            self.rrt_preds = preds[-1] # (num_prefs,)

        if only_out:
            self.act_labels = labels[1] # (num_prefs, window_size)

            # Binary or Multi-Class Outcome (BO or MCO) prediction, labels 
            # BO: shape (num_prefs, 1) or (num_prefs_out, 1), dtype torch.float32 
            # MCO: shape (num_prefs,) or (num_prefs_out,), dtype torch.int64 
            self.out_labels = labels[2] 

            # Binary or Multi-Class Outcome (BO or MCO) prediction, predictions 
            # BO: shape (num_prefs,) or (num_prefs_out,), dtype torch.float32 
            # MCO: shape (num_prefs, num_outclasses) or (num_prefs_out, num_outclasses), dtype torch.float32 
            self.out_pred = preds[-1] 
        if both:
            self.rrt_labels = labels[1] # (num_prefs, window_size, 1)
            # Selecting the RRT labels corresponding to the first 
            # decoding step only. 
            self.rrt_labels = self.rrt_labels[:, 0, 0] # (num_prefs,)
            self.act_labels = labels[2] # (num_prefs, window_size)

            # Binary or Multi-Class Outcome (BO or MCO) prediction, labels 
            # BO: shape (num_prefs, 1) or (num_prefs_out, 1), dtype torch.float32 
            # MCO: shape (num_prefs,) or (num_prefs_out,), dtype torch.int64 
            self.out_labels = labels[3] 

            # Direct RRT predictions (still in standardized scale)
            self.rrt_preds = preds[-2] # (num_prefs,)

            # Binary or Multi-Class Outcome (BO or MCO) prediction, predictions 
            # BO: shape (num_prefs,) or (num_prefs_out,), dtype torch.float32 
            # MCO: shape (num_prefs, num_outclasses) or (num_prefs_out, num_outclasses), dtype torch.float32 
            self.out_pred = preds[-1] 

        if both_not:
            self.act_labels = labels[-1] # (num_prefs, window_size)

        self.remaining_runtime_head = remaining_runtime_head
        self.outcome_bool = outcome_bool


        # window_size corresponds to the maximum sequence length of 
        # the prefix and suffixes. This is hence also the size of the 
        # sequence length dimension of all prefix and suffix tensors, 
        # which are all right padded for shorter prefixes and suffixes. 
        self.window_size = self.act_labels.shape[-1]
        # `self.batch_size` corresponds to the number of instances aka 
        # the number of prefix-suffix pairs, aka 'num_prefs'. 
        self.batch_size = self.act_labels.shape[0]

        # End Token gets last index, and should be contained 
        # within the label tensor for all of the prefix-suffix 
        # pairs. That's how we derive the num_classes from it. 
        self.num_classes = torch.max(self.act_labels).item() + 1

        self.mean_std_ttne = mean_std_ttne
        self.mean_std_tsp = mean_std_tsp
        self.mean_std_tss = mean_std_tss
        self.mean_std_rrt = mean_std_rrt

        # Updating suffix_acts_decoded such that only the first END 
        # token prediction is retained, and everything after it padded, 
        # and retrieving the predicted suffix_length of all instances.
        self.pred_length = self.edit_suffix_acts()

        self.actual_length = self.get_actual_length()

        # Converting the ttne suffix and rrt labels from standardized back into original scale. 
        self.ttne_labels_seconds = self.convert_to_seconds(time_string='ttne', input_tensor=self.ttne_labels.clone()) # (num_prefs, window_size)
        self.rrt_labels_seconds = self.convert_to_seconds(time_string='rrt', input_tensor=self.rrt_labels.clone()) # (num_prefs,)



    def edit_suffix_acts(self):
        """For each of the batch_size instances, replace all predictions 
        after the first END token prediction with padding values (idx 0). 

        Also compute and return the predicted suffix length (0-based), 
        i.e. the index at which the model predicted the END token. 
        """
        # Initialize tensor that contains 1 where end token 
        # (idx num_classes-1) is predicted, 0 otherwise. 
        end_pred_tens = (self.suffix_acts_decoded== (self.num_classes-1)).to(torch.int64) # (batch_size, window_size)
        # Init tensor that contains for each instance the number of times 
        # an end token is predicted 
        sum_tens = torch.sum(input=end_pred_tens, dim=-1) # (batch_size,)

        # Artificial end token inserted on the last time step, for all 
        # instances for which the model never produced an END prediction
        suffix_acts_dec_help = self.suffix_acts_decoded.clone() # (batch_size, window_size)
        suffix_acts_dec_help[:,-1] = self.num_classes-1
        # (batch_size, window_size)
        self.suffix_acts_decoded = torch.where(condition=(sum_tens==0).unsqueeze(-1), input=suffix_acts_dec_help, other=self.suffix_acts_decoded)

        # Right padding decoded activity predictions with index 0 
        # starting from the first END token prediction
        pred_length = torch.argmax((self.suffix_acts_decoded== (self.num_classes-1)).to(torch.int64), dim=-1) 
        counting_tensor = torch.arange(self.window_size, dtype=torch.int64).to(device) # (window_size,)
        #       Repeat the tensor along the first dimension to match the desired shape
        counting_tensor = counting_tensor.unsqueeze(0).repeat(self.batch_size, 1).to(device) # (batch_size, window_size)
        padding_bool = counting_tensor > pred_length.unsqueeze(-1) # (batch_size, window_size)
        padding_inds = torch.nonzero(input=padding_bool, as_tuple=True)
        self.suffix_acts_decoded[padding_inds] = 0

        return pred_length

    def get_actual_length(self):
        """Get the ground truth suffix length of each of the batch_size 
        instances. Computed based on the act_labels, by determining 
        the suffix index where the ground truth 'END TOKEN' is located. 
        Note: this is 0-based. E.g., a value of 0 would indicate that the 
        END TOKEN should be predicted immediately at the very first 
        decoding step. 
        """
        actual_length = torch.argmax((self.act_labels == (self.num_classes-1)).to(torch.int64), dim=-1) # (batch_size,)
        return actual_length

    def convert_to_seconds(self, time_string, input_tensor):
        """Convert a tensor of any shape, containing time-related 
        features, predictions or labels, that are standardized, back 
        into the original scale of seconds. Negative values are clamped
        to zero to prevent negative times.

        Parameters
        ----------
        time_string : {'ttne', 'tsp', 'tss', 'rrt'}
            String indicating which type of time feature / target / 
            prediction needs to be converted. 
        input_tensor : torch.Tensor
            Tensor to be converted into the original seconds scale. 
        """
        if time_string == 'ttne':
            train_mean = self.mean_std_ttne[0]
            train_std = self.mean_std_ttne[1]
        elif time_string == 'tsp':
            train_mean = self.mean_std_tsp[0]
            train_std = self.mean_std_tsp[1]
        elif time_string == 'tss':
            train_mean = self.mean_std_tss[0]
            train_std = self.mean_std_tss[1]
        elif time_string == 'rrt':
            train_mean = self.mean_std_rrt[0]
            train_std = self.mean_std_rrt[1]
        
        converted_tensor = input_tensor*train_std + train_mean
        converted_tensor = torch.clamp(converted_tensor, min=0)

        return converted_tensor # Same shape as input_tensor

    def convert_to_transf(self, time_string, input_tensor):
        """Convert a tensor of any shape, containing time-related 
        features, predictions or labels, that are expressed in seconds, 
        back into the transformed scale, which is obtained by 
        standardization.

        Parameters
        ----------
        time_string : {'ttne', 'tsp', 'tss', 'rrt'}
            String indicating which type of time feature / target / 
            prediction needs to be converted. 
        input_tensor : torch.Tensor
            Tensor to be converted back into the transformed scale. 
        """
        if time_string == 'ttne':
            train_mean = self.mean_std_ttne[0]
            train_std = self.mean_std_ttne[1]
        elif time_string == 'tsp':
            train_mean = self.mean_std_tsp[0]
            train_std = self.mean_std_tsp[1]
        elif time_string == 'tss':
            train_mean = self.mean_std_tss[0]
            train_std = self.mean_std_tss[1]
        elif time_string == 'rrt':
            train_mean = self.mean_std_rrt[0]
            train_std = self.mean_std_rrt[1]
        
        converted_tensor = (input_tensor - train_mean) / train_std

        return converted_tensor # Same shape as input_tensor

    def compute_ttne_results(self):
        """Compute the absolute errors for the timestamp (Time Till Next 
        Event, aka TTNE) suffix predictions in both the standardized 
        (~N(0,1)) and the original (in seconds) scale. 

        Each of the `num_prefs` prefix-suffix pairs 
        (inference instances), the TTNE suffix is comprised of a sequence 
        of `window_size` TTNE values, both the TTNE predictions and the 
        TTNE labels. Given one particular inference instance, its 
        `window_size` absolute differences are hence computed by 
        taking the absolute value of the difference between each  
        ground-truth TTNE value, and the corresponding prediction. For 
        the majority of the cases however, a significant portion of the 
        `window_size` TTNE values are (right-)padded values, and hence do 
        not correspond to actual suffix events. Therefore, only the 
        absolute differences pertaining to actual (ground-truth) suffix 
        events are selected and returned. 
        
        Furthermore, to resemble real-
        life situations, in which timestamp predictions for positions 
        after the position at which the END token (activity suffix) is 
        predicted are ignored, as closely as possible, the TTNE 
        predictions after the index of the predicted END token are 
        replaced with a zero(-equivalent). Additionally, since decoded 
        activity labels fed to the decoder after the prediction of the 
        END token do not contain any informational value, these 
        timestamp predictions would not make any sense anyway. 
        
        Returns
        -------
        MAE_ttne_stand : torch.Tensor 
            NOTE: the returned tensor has shape (num_prefs, window_size), 
            instead of the shape specified in the explanation below. 
            The subsetting of the absolute errors pertaining to the 
            actual non-padded suffix events, as explained below, 
            will be done inside the functions calling this method. 

            Legacy explanation:
            Absolute errors standardized timestamp predictions for all 
            actual / non-padded suffix events. Shape (NP,), with NP being 
            equal to the total number of actual suffix events (including 
            the added END tokens) over all `num_prefs` (aka `batch_size`) 
            prefix-suffix pairs in the validation or test set. 
        MAE_ttne_seconds : torch.Tensor 
            NOTE: the returned tensor has shape (num_prefs, window_size), 
            instead of the shape specified in the explanation below. 
            The subsetting of the absolute errors pertaining to the 
            actual non-padded suffix events, as explained below, 
            will be done inside the functions calling this method. 
            
            Legacy explanation:
            The absolute errors pertaining to the same predictions as the 
            errors contained within `MAE_ttne_stand`, but after first 
            having converted their respective predictions and labels back 
            into the original scale (in seconds). (This is done based on 
            the training mean and standard deviation of the actual / non-
            padded TTNE values, contained within `self.mean_std_ttne`.)
        """
        
        # De-standardizing the decoded predictions
        self.suffix_ttne_preds_seconds = self.convert_to_seconds(time_string='ttne', input_tensor=self.suffix_ttne_preds.clone())
        # self.suffix_ttne_preds_seconds = self.suffix_ttne_preds_seconds*self.mean_std_ttne[1] + self.mean_std_ttne[0] # (batch_size, window_size)
        
        # Defining the 0-equivalent value for the standardized predictions and labels
        stand_0_eq = -self.mean_std_ttne[0]/self.mean_std_ttne[1]


        # --------------------------------------------
        #       Generate a tensor with values counting from 0 to window_size
        counting_tensor = torch.arange(self.window_size, dtype=torch.int64).to(device) # (window_size,)
        #       Repeat the tensor along the first dimension to match the desired shape
        counting_tensor = counting_tensor.unsqueeze(0).repeat(self.batch_size, 1).to(device) # (batch_size, window_size)

        # Deriving boolean tensor of shape (batch_size, window_size) 
        # containing True for the indices after its END token prediction 
        # in the predicted activity suffix 
        pad_preds = counting_tensor > self.pred_length.unsqueeze(-1) # (batch_size, window_size)

        # Padding both prediction tensors 
        self.suffix_ttne_preds_seconds[pad_preds] = 0 # (batch_size, window_size)
        self.suffix_ttne_preds[pad_preds] = stand_0_eq # (batch_size, window_size)

        # Deriving boolean tensor of shape (batch_size, window_size) with 
        # True on the indices before and on the ground truth END token 
        # NOTE: change May 11 -> Will only be done afterwards, the slicing. 
        # before_end_token = counting_tensor <= self.actual_length.unsqueeze(-1)

        # Computing MAE ttne elements and slicing out the considered ones 
        MAE_ttne_stand = torch.abs(self.suffix_ttne_preds - self.ttne_labels) # (batch_size, window_size)
        # MAE_ttne_stand = MAE_ttne_stand[before_end_token] # shape (torch.sum(self.actual_length+1), )

        MAE_ttne_seconds = torch.abs(self.suffix_ttne_preds_seconds - self.ttne_labels_seconds) # (batch_size, window_size)
        # MAE_ttne_seconds = MAE_ttne_seconds[before_end_token] # shape (torch.sum(self.actual_length+1), )

        # return MAE_1_stand, MAE_1_seconds, MAE_2_stand, MAE_2_seconds

        # NOTE: change 11/05: Slicing out is only done afterwards, this now returns 
        # two (batch_size, window_size) shaped tensors. 
        return MAE_ttne_stand, MAE_ttne_seconds
    

    def compute_rrt_results(self):
        """Compute MAE for the remaining runtime predictions, in the 
        preprocessed (standardized, ~N(0,1)) scale, as well as in the 
        original scale (seconds) by de-standardizing based on the 
        training mean and standard deviation used for standardizing the 
        RRT (Remaining RunTime) target. 
        """
        # De-standardizing rrt predictions to scale in seconds based on 
        # training mean and standard deviation of the RRT labels
        rrt_preds_seconds = self.convert_to_seconds(time_string='rrt', input_tensor=self.rrt_preds.clone()) # (num_prefs,)

        # Computing absolute errors as-is, and in seconds. 
        abs_errors = torch.abs(self.rrt_preds - self.rrt_labels) # (num_prefs,)
        abs_errors_seconds = torch.abs(rrt_preds_seconds - self.rrt_labels_seconds) # (num_prefs,)

        return abs_errors, abs_errors_seconds # both (batch_size,)




    def damerau_levenshtein_distance_tensors(self):
        """Compute the (normalized) damerau-levenshtein distance for each of 
        the `batch_size` predicted suffixes in parallel. 

        Notes
        -----
        Leverages the following `BatchInference` attributes:

        * `self.suffix_acts_decoded` : torch.Tensor containing the 
          integer-encoded sequence of predicted activities. Shape 
          (self.batch_size, self.window_size) and dtype torch.int64. The 
          predicted suffix activities are derived greedily, by taking 
          the activity label index pertaining to the highest probability 
          in `self.act_preds`. 

        * `self.act_labels` : torch.Tensor containing the integer-encoded 
          sequence of ground-truth activity label suffixes. Shape 
          (self.batch_size, self.window_size) and dtype torch.int64. 
        
        * `self.pred_length` : torch.Tensor of dtype torch.int64 and 
          shape (self.batch_size, ). Contains the (0-based) index at 
          which the model predicted the END token during decoding, for 
          each of the self.batch_size instances. Consequently, for each 
          instance i (i=0, ..., self.batch_size-1), the predicted suffix 
          length is equal to self.pred_length[i] + 1. 
        
        * `self.actual_length` : torch.Tensor of dtype torch.int64 and 
          shape (self.batch_size, ). Contains the (0-based) index of the 
          END token in the ground-truth activity suffix for each of the 
          (self.batch_size,) instances. Consequently, for each instance i 
          (i=0, ..., self.batch_size-1), the ground-truth suffix length 
          is equal to self.actual_length[i] + 1. 

        Returns
        -------
        dam_lev_dist : torch.Tensor
            Tensor containing the normalized Damerau-Levenshtein distance 
            for each of the `batch_size` prefix-suffix pairs / instances. 
            Shape (batch_size, ), dtype torch.float32. 
        """
        len_pred = self.pred_length+1 # (self.batch_size, )
        len_actual = self.actual_length+1 # (self.batch_size, )
        # max_length between the ground-truth and predicted activity sequence
        max_len = torch.maximum(len_pred, len_actual) # (self.batch_size,)

        # Initializing the (self.batch_size, self.window_size+1, 
        # self.window_size+1)-shaped distance matrix, with the innermost 
        # dimension representing the activity labels, the central 
        # dimension the predicted activities, and the outermost dimension 
        # the self.batch_size instances. (self.window_size+1 because 
        # first row and column stand for empty strings)
        d = torch.full(size=(self.batch_size, self.window_size+1, self.window_size+1), fill_value=0, dtype=torch.int64).to(device) # (B, WS+1, WS+1)

        # Initialize distances first row and column for each of the 
        # self.batch_size instances (empty strings)
        arange_tens = torch.arange(start=0, end=self.window_size+1, dtype=torch.int64).unsqueeze(0).to(device) # (1, WS+1)
        # First row for each instance 
        d[:, 0, :] = arange_tens
        # First column for each instance 
        d[:, :, 0] = arange_tens

        # Outer loop over predicted sequence (rows) for all of the instances
        # Note that in this loop, both the index for the rows and the 
        # index for the columns refer to the previous letters in both 
        # words. I.e. index 1 refers to 0th letter (1st, 0-based).
        for i in range(1, self.window_size+1): 
            # Inner loop over the ground-truth sequence (columns) for all of the instances
            for j in range(1, self.window_size+1):
                # At each position, make a (self.batch_size, ) shaped 
                # tensor containing the integer distances for each of the 
                # 4 possible operations. Then derive the minimum cost 
                # into a integer tensor 'min_cost' of shape 
                # (self.batch_size, ). Then d[:, i, j] = min_cost. 
                
                # Get (self.batch_size, )-shaped cost tensor for the 
                # substitution cost (0 or 1)
                cost = torch.where(self.suffix_acts_decoded[:, i-1]==self.act_labels[:, j-1], 0, 1) 
                
                # Get (self.batch_size, )-shaped distances for the 
                # respective cell, in case of deletion.
                deletion = d[:, i-1, j] + 1 # (self.batch_size, )

                # Get (self.batch_size, )-shaped distances for the 
                # respective cell, in case of insertion.
                insertion = d[:, i, j-1] + 1 # (self.batch_size, )

                # Get (self.batch_size, )-shaped distances for the 
                # respective cell, in case of substitution.
                substition = d[:, i-1, j-1] + cost # (self.batch_size, )

                # Update distance respective cell based on the 
                # cheapest option (deletion, insertion, substitution)
                d[:, i, j] = torch.minimum(torch.minimum(deletion, insertion), substition)

                # Check whether transposition would be cheaper. 
                if i > 1 and j > 1:
                    # Derive boolean tensor of shape (batch_size,), with 
                    # True indicating transposition is possible for 
                    # that respective instance. False if not. 
                    tpos_true = (self.suffix_acts_decoded[:, i-1]==self.act_labels[:, j-2]) & (self.suffix_acts_decoded[:, i-2]==self.act_labels[:, j-1])

                    # Computing minimum cost between 
                    # min(deletion, insertion, substitution), and 
                    # superposition, for all instances, even for the ones 
                    # for which superposition would not be possible. 
                    min_og_tpos = torch.minimum(d[:, i, j], d[:, i-2, j-2]+cost) # (self.batch_size, )

                    # Updating distance respective cell for those 
                    # instances for which transposition is possible and 
                    # cheaper. 
                    d[:, i, j] = torch.where(tpos_true, min_og_tpos, d[:, i, j])

        # Derive integer indexing tensor for the outermost (batch) 
        # dimension
        batch_arange = torch.arange(start=0, end=self.batch_size, dtype=torch.int64).to(device) # (self.batch_size, )

        # For each of the batch_size instances i, the dam lev distance is 
        # contained in the cell with row index len_pred[i] and column 
        # index len_actual[i]. Normalize by the longest distance between 
        # predicted and ground-truth suffix for each of the instances too.
        dam_lev_dist = d[batch_arange, len_pred, len_actual] / max_len  # (self.batch_size, )
        
        return dam_lev_dist
    
    def compute_suf_length_diffs(self):
        """Compute different measures concerning the difference between 
        the predicted and ground-truth suffix lengths. 
        """
        too_early_bool = self.pred_length < self.actual_length # (batch_size,)
        too_late_bool = self.pred_length > self.actual_length # (batch_size, )
        length_diff = self.pred_length - self.actual_length # (batch_size,)
        length_diff_too_early = length_diff[too_early_bool].clone() 
        length_diff_too_late = length_diff[too_late_bool].clone()

        amount_right = torch.sum(length_diff == 0) # scalar tensor 

        return length_diff, length_diff_too_early, length_diff_too_late, amount_right

    def compute_outcome_BCE(self):
        """Compute Binary Cross Entropy for the binary outcome prediction 
        task. 
        """
        # initializing loss computation 
        criterion_nonred = nn.BCELoss(reduction='none')
        with torch.no_grad():
            loss_nonred = criterion_nonred(self.out_pred.unsqueeze(-1), self.out_labels)[:, 0] # (batch_size,)
        
        return loss_nonred
    
    
    def compute_AUC(self):
        """Computes the AUC-ROC and AUC-PR for binary outcome prediction. 
        """
        # Converting the tensors to numpy arrays 
        labels_np = self.out_labels[:, 0].numpy() # (num_prefs,)
        preds_np = self.out_pred.numpy() # (num_prefs,)

        # Computing AUC ROC 
        auc_roc = roc_auc_score(labels_np, preds_np)

        # Computing AUC PR 
        precision, recall, thresholds = precision_recall_curve(labels_np, preds_np)
        auc_pr = auc(recall, precision)

        return auc_roc, auc_pr
    

    def compute_AUC_CaseBased(self, sample_weight):
        """Compute CaLenDiR's Case-Based AUC-ROC and AUC-PR scores for 
        binary outcome prediction.

        Parameters
        ----------
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
        labels_np = self.out_labels[:, 0].numpy() # (num_prefs,) or (num_prefs_out,)
        preds_np = self.out_pred.numpy() # (num_prefs,) or (num_prefs_out,)

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

    def compute_binary_metrics_CaseBased(self, 
                                         sample_weight: torch.Tensor,
                                         threshold: float = 0.5):
        """
        Compute accuracy, F1, precision, recall for binary outcome (BO) 
        prediction.
        
        Attributes Used
        ---------------
        self.out_pred : torch.Tensor
            Shape (batch_size,), contains predicted probabilities
            for the positive class (already sigmoid'ed).
            Dtype torch.float32
        self.out_labels : torch.Tensor
            Shape (batch_size,) or (batch_size,1), ground truth labels (0 or 1).
            Dtype torch.float32 or torch.int64.

        Parameters
        ----------
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
        labels_flat = self.out_labels.view(-1).long()  # shape (T,)
        
        # 2. Convert probabilities to predicted labels using threshold
        pred_binary = (self.out_pred >= threshold).long()  # shape (T,)
        
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

    def compute_binary_metrics(self, 
                               threshold: float = 0.5):
        """
        Compute accuracy, F1, precision, recall for binary classification.
        
        Attributes Used
        ---------------
        self.out_pred : torch.Tensor
            Shape (batch_size,), contains predicted probabilities
            for the positive class (already sigmoid'ed).
            Dtype torch.float32
        self.out_labels : torch.Tensor
            Shape (batch_size,) or (batch_size,1), ground truth labels (0 or 1).
            Dtype torch.float32 or torch.int64.

        Parameters
        ----------
        threshold : float, optional
            Probability threshold to decide positive/negative.
        
        Returns
        -------
        dict
            Dictionary with accuracy, f1, precision, recall,
            and optionally balanced_accuracy.
        """
        
        # 1. Flatten labels if needed and convert to int if they're floats
        labels_flat = self.out_labels.view(-1).long()  # shape (T,)
        
        # 2. Convert probabilities to predicted labels using threshold
        pred_binary = (self.out_pred >= threshold).long()  # shape (T,)
        
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



    # Multi-Class Outcome Prediction functions 
    # def compute_multiclass_metrics(out_pred_global: torch.Tensor, 
    #                             out_labels: torch.Tensor):
    def compute_multiclass_metrics(self):
        """Compute accuracy, macro-, and weighted-F1, Precision and Recall, 
        for multi-class outcome (MCO) prediction. 
        
        Attributes Used
        ---------------
        self.out_pred : torch.Tensor
            shape (batch_size, num_outclasses), unnormalized logits. 
            No softmax is applied yet. Dtype torch.float32
        self.out_labels : torch.Tensor
            shape (batch_size,), ground truth class indices. Dtype 
            torch.int64. 
        """
        
        # 1. Get predicted class by taking argmax over logits
        pred_classes = torch.argmax(self.out_pred, dim=1) # shape (batch_size,), torch.int64
        
        # 2. Convert to numpy arrays (if needed) for sklearn
        pred_classes_np = pred_classes.cpu().numpy()
        labels_np = self.out_labels.cpu().numpy()
        
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



    # def compute_multiclass_metrics_CaseBased(out_pred_global: torch.Tensor, 
    #                                         out_labels: torch.Tensor, 
    #                                         sample_weight : torch.Tensor):
    def compute_multiclass_metrics_CaseBased(self, sample_weight : torch.Tensor):
        """Compute CaLenDiR's Case-Based  accuracy, macro-, and weighted-F1, 
        Precision and Recall, for multi-class outcome (MCO) prediction. 
        
        Attributes Used
        ---------------
        self.out_pred : torch.Tensor
            shape (batch_size, num_outclasses), unnormalized logits. 
            No softmax is applied yet. Dtype torch.float32
        self.out_labels : torch.Tensor
            shape (batch_size,), ground truth class indices. Dtype 
            torch.int64. 
        
        Parameters
        ----------
        sample_weight : torch.Tensor 
            Tensor of dtype torch.float32 and shape (batch_size,). 
            It contains for each of the (subsetted) instances 
            the weight it should be given for each original test set case 
            to contribute equally. 
        """
        sample_weight = sample_weight.numpy() # (num_prefs,) or (num_prefs_out,)

        # 1. Get predicted class by taking argmax over logits
        pred_classes = torch.argmax(self.out_pred, dim=1) # shape (batch_size,), torch.int64
        
        # 2. Convert to numpy arrays (if needed) for sklearn
        pred_classes_np = pred_classes.cpu().numpy()
        labels_np = self.out_labels.cpu().numpy()
        
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


    def compute_MCO_CE(self):
        """Compute the masked Categorical Cross-Entropy (CCE) inference 
        metric for Multi-Class Outcome (MCO) prediction. 

        Attributes Used
        ---------------
        self.out_pred : torch.Tensor
            shape (num_prefs, num_outclasses), unnormalized logits. 
            No softmax is applied yet. Dtype torch.float32. (In case of 
            an outcome mask, num_prefs becomes num_prefs_out.)
        self.out_labels : torch.Tensor
            shape (num_prefs,), ground truth class indices. Dtype 
            torch.int64. (In case of 
            an outcome mask, num_prefs becomes num_prefs_out.)
        """
        criterion_nonred = nn.CrossEntropyLoss(reduction='none')
        with torch.no_grad():
            loss_nonred = criterion_nonred(self.out_pred, self.out_labels) # (num_prefs,)

        return loss_nonred 