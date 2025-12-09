

import torch
import torch.nn as nn
import numpy as np
# 'uncomment' device setup line underneath for leveraging GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# comment out line underneath for leveraging GPU
device = torch.device("cpu")


class BatchInference():
    def __init__(self,
                 preds,
                 labels, 
                 seq_task,
                 mean_std_ttne=None, 
                 mean_std_tsp=None, 
                 mean_std_tss=None):
        """Init BatchInference object for the SuTraN_SST model. Modified 
        to compute the metrics over the predictions made for the whole 
        inference (validation or test) set at once. Therefore, to ensure 
        the satisfaction of possible memory constraints and given the 
        fact that these computations should only be performed once, 
        instead of for every individual batch, all tensors and 
        computations are stored and performed on the CPU, instead of 
        the GPU. 

        Contains methods for computing the different 
        evaluation metrics for activity suffix and timestamp suffix 
        prediction, made by the single-task SuTraN_SST variant. 

        Preprocesses both predictions and labels in different scales 
        and representations, upon which the methods for computing the 
        metrics can be called. 

        Parameters
        ----------
        preds : torch.Tensor 
            Tensor containing either the activity or timestamp suffix 
            predictions made by the Sequential, Single-Task (SST) SuTraN 
            variant, depending on the `seq_task` parameter. In both 
            cases, the `preds` tensor is of shape 
            (num_prefs, window_size). It is of the torch.int64 dtype in 
            case `seq_task='activity_suffix'`, and of the torch.float32 
            dtype in case `seq_task='timestamp_suffix'`. 
        labels : torch.Tensor
            Tensor containing the ground-truth activity suffix labels in 
            case of `seq_task='activity_suffix'` (shape 
            (num_prefs, window_size) and dtype torch.int64), or the 
            ground-truth timestamp suffix labels in case of 
            `seq_task='timestamp_suffix'` (shape 
            (num_prefs, window_size, 1) and dtype torch.float32). 
            In the latter case, the trailing dimension of size one will 
            immediately dropped upon initialization, resulting in again a 
            tensor of shape (num_prefs, window_size). 
        seq_task : {'activity_suffix', 'timestamp_suffix'}
            The (sole) sequential prediction task trained and evaluated 
            for. 
        mean_std_ttne : None or list of float, optional
            Training mean and standard deviation used to standardize the time 
            till next event (in seconds) target. Needed for re-converting 
            ttne predictions to original scale. Mean is the first entry, 
            std the second.
            Only needed when `seq_task='timestamp_suffix'`. The default 
            value of `None` can be retained otherwise. 
        mean_std_tsp : None or list of float, optional
            Training mean and standard deviation used to standardize the time 
            since previous event (in seconds) feature of the decoder suffix 
            tokens. Needed for re-converting time since previous event values 
            to original scale (seconds). Mean is the first entry, std the 2nd.
            Only needed when `seq_task='timestamp_suffix'`. The default 
            value of `None` can be retained otherwise. 
        mean_std_tss : None or list of float, optional
            Training mean and standard deviation used to standardize the time 
            since start (in seconds) feature of the decoder suffix 
            tokens. Needed for re-converting time since start values 
            to original scale (seconds). Mean is the first entry, std the 2nd.
            Only needed when `seq_task='timestamp_suffix'`. The default 
            value of `None` can be retained otherwise. 

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
        self.act_sufbool = (seq_task=='activity_suffix') 
        self.ts_sufbool = (seq_task=='timestamp_suffix')
        if not (self.act_sufbool or self.ts_sufbool):
            raise ValueError(
                "`seq_task={}` is not a valid argument. It ".format(seq_task) + 
                "should be either `'activity_suffix'` or "
                "`'timestamp_suffix'`."
            )

        self.mean_std_ttne = mean_std_ttne
        self.mean_std_tsp = mean_std_tsp
        self.mean_std_tss = mean_std_tss

        if self.ts_sufbool: 
            self.ttne_labels = labels # (num_prefs, window_size, 1)
            # Drop trailing dimension of size 1 
            self.ttne_labels = self.ttne_labels[:, :, 0] # (num_prefs, window_size)
            self.suffix_ttne_preds= preds # (batch_size, window_size), torch.float32

            # window_size corresponds to the maximum sequence length of 
            # the prefix and suffixes. This is hence also the size of the 
            # sequence length dimension of all prefix and suffix tensors, 
            # which are all right padded for shorter prefixes and suffixes. 
            self.window_size = self.ttne_labels.shape[-1]
            # `self.batch_size` corresponds to the number of instances aka 
            # the number of prefix-suffix pairs, aka 'num_prefs'. 
            self.batch_size = self.ttne_labels.shape[0]

            # Converting the ttne suffix and rrt labels from standardized back into original scale. 
            self.ttne_labels_seconds = self.convert_to_seconds(time_string='ttne', input_tensor=self.ttne_labels.clone()) # (num_prefs, window_size)

        
        elif self.act_sufbool:
            self.act_labels = labels # (num_prefs, window_size), torch.int64
            #   - Greedily predicted activity suffixes for all validation / 
            #     test instances 
            self.suffix_acts_decoded = preds # (num_prefs, window_size)

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


        if self.act_sufbool: 
            # Updating self.suffix_acts_decoded such that only the first 
            # END token prediction is retained, and everything after it 
            # padded, and retrieving the predicted suffix_length of all 
            # instances.
            self.pred_length = self.edit_suffix_acts()

        # Determining actual ground-truth suffix length
        self.actual_length = self.get_actual_length()



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
        instances. Computed based on the act_labels or ttne_labels.
        Note: this is 0-based. E.g., a value of 0 would indicate that the 
        END TOKEN should be predicted immediately at the very first 
        decoding step. 
        """
        if self.act_sufbool:
            actual_length = torch.argmax((self.act_labels == (self.num_classes-1)).to(torch.int64), dim=-1) # (batch_size,)
        
        elif self.ts_sufbool: 
            actual_length = torch.argmax((self.ttne_labels == -100).to(torch.int64), dim=-1) # (batch_size,)
            actual_length = actual_length - 1 
            # Correcting for instances in which ground-truth suffix length equals max 
            # window_size, and hence in which no padded values of -100 were found in the 
            # ttne labels 
            actual_length = torch.where(actual_length == -1, self.window_size-1, actual_length)
        return actual_length
    

    def convert_to_seconds(self, time_string, input_tensor):
        """Convert a tensor of any shape, containing time-related 
        features, predictions or labels, that are standardized, back 
        into the original scale of seconds. 

        Parameters
        ----------
        time_string : {'ttne', 'tsp', 'tss'}
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
        else:
            raise ValueError(f"Unsupported time_string '{time_string}'")
        
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
        time_string : {'ttne', 'tsp', 'tss'}
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
        else:
            raise ValueError(f"Unsupported time_string '{time_string}'")
        
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

            Updated explanation:
            Absolute errors between standardized predictions and labels,
            shape `(num_prefs, window_size)`. Callers slice out padded
            positions before aggregating.
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

            Updated explanation:
            Same absolute errors expressed in seconds, also
            `(num_prefs, window_size)`. Callers slice out padded
            positions before aggregating.
        """
        
        # De-standardizing the decoded predictions
        self.suffix_ttne_preds_seconds = self.convert_to_seconds(time_string='ttne', input_tensor=self.suffix_ttne_preds.clone())        

        # Computing MAE ttne elements and slicing out the considered ones
        MAE_ttne_stand = torch.abs(self.suffix_ttne_preds - self.ttne_labels) # (batch_size, window_size)
        # MAE_ttne_stand = MAE_ttne_stand[before_end_token] # shape (torch.sum(self.actual_length+1), )

        MAE_ttne_seconds = torch.abs(self.suffix_ttne_preds_seconds - self.ttne_labels_seconds) # (batch_size, window_size)
        # MAE_ttne_seconds = MAE_ttne_seconds[before_end_token] # shape (torch.sum(self.actual_length+1), )

        # NOTE: Slicing out is only done afterwards, this now returns 
        # two (batch_size, window_size) shaped tensors. 
        return MAE_ttne_stand, MAE_ttne_seconds

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