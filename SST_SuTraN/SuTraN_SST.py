"""
Code of the Sequential, Single-Task (SST) version of SuTraN for the
sequential prediction tasks, being activity suffix or timestamp suffix
prediction.
"""

import torch 
import torch.nn as nn
import math

from SuTraN.transformer_prefix_encoder import EncoderLayer
from SuTraN.transformer_suffix_decoder import DecoderLayer

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    """Inject sequence information in the prefix or suffix embeddings 
    before feeding them to the stack of encoders or decoders respectively. 

    Predominantly based on the PositionalEncoding module defined in 
    https://github.com/pytorch/examples/tree/master/word_language_model. 
    This reimplemetation, in contrast to the original one, caters for 
    adding sequence information in input embeddings where the batch 
    dimension comes first (``batch_first=True`). 

    Parameters
    ----------
    d_model : int
        The embedding dimension adopted by the associated Transformer. 
    dropout : float
        Dropout value. Dropout is applied over the sum of the input 
        embeddings and the positional encoding vectors. 
    max_len : int
        the max length of the incoming sequence. By default 10000. 
    """


    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()

        # Check if d_model is an integer and is even
        assert isinstance(d_model, int), "d_model must be an integer"
        assert d_model % 2 == 0, "d_model must be an even number"
        
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) 
        self.register_buffer('pe', pe) # shape (max_len, d_model)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Sequence of prefix event tokens or suffix event tokens fed 
            to the positional encoding module. Shape 
            (batch_size, window_size, d_model).

        Returns
        -------
        Updated sequence tensor of the same shape, with sequence 
        information injected into it, and dropout applied. 
        """
        x = x + self.pe[:x.size(1), :] # (batch_size, window_size, d_model)
        return self.dropout(x)

class SuTraN_SST(nn.Module):
    def __init__(self, 
                 num_activities, 
                 d_model, 
                 cardinality_categoricals_pref, 
                 num_numericals_pref, 
                 seq_task, 
                 num_prefix_encoder_layers=4, 
                 num_decoder_layers=4,
                 num_heads=8, 
                 d_ff=128, 
                 dropout=0.2, 
                 layernorm_embeds=True
                 ):
        """Initialize an instance of the Sequential, Single-Task 
        (SST), version of SuTraN. To be trained for 
        one of the two Sequential prediction targets, specified by 
        the `seq_task` argument. 

        #. Activity Suffix prediction
        #. Timestamp Suffix prediction
        
        
        The learned activity embedding weight matrix is shared between 
        the encoder and decoder. 

        Parameters
        ----------
        num_activities : int
            Number of distinct activities present in the event log. 
            This does include the end token and padding token 
            used for the activity labels. For the categorical activity 
            label features in the prefix and suffix
            (in case `seq_task='activity_suffix'`), no END token is
            included. Hence, the amount of distinct levels there is 
            equal to `num_activities`-1. 
        d_model : int
            Model dimension. Each sublayer of the encoder and decoder 
            blocks take as input a (batch_size, window_size, d_model) 
            shaped tensor, and output an updated tensor of the same 
            shape. 
        cardinality_categoricals_pref : list of int
            List of `num_categoricals` integers. Each integer entry 
            i (i = 0, ..., `num_categoricals`-1) contains the cardinality 
            of the i'th categorical feature of the encoder prefix events. 
            The order of the cardinalities should match the order in 
            which the categoricals are fed as inputs. Note that for each 
            categorical, an extra category should be included to account 
            for missing values.
        num_numericals_pref : int 
            Number of numerical features of the prefix events
        seq_task : {'activity_suffix', 'timestamp_suffix'}
            The (sole) sequential prediction task trained and evaluated 
            for. 
        num_prefix_encoder_layers : int, optional
            The number of prefix encoder blocks stacked on top of each 
            other, by default 4.
        num_decoder_layers : int, optional
            Number of decoder blocks stacked on top of each other, 
            by default 4.
        num_heads : int, optional
            Number of attention heads for the Multi-Head Attention 
            sublayers in both the encoder and decoder blocks, by default 
            8.
        d_ff : int, optional
            The dimension of the hidden layer of the point-wise feed 
            forward sublayers in the transformer blocks, by default 128.
        dropout : float, optional
            Dropout rate during training. By default 0.2. 
        layernorm_embeds : bool, optional
            Whether or not Layer Normalization is applied over the 
            initial embeddings of the encoder and decoder. True by 
            default.
        """
        super(SuTraN_SST, self).__init__()
        self.act_sufbool = (seq_task=='activity_suffix') 
        self.ts_sufbool = (seq_task=='timestamp_suffix')
        if not (self.act_sufbool or self.ts_sufbool):
            raise ValueError(
                "`seq_task={}` is not a valid argument. It ".format(seq_task) + 
                "should be either `'activity_suffix'` or "
                "`'timestamp_suffix'`."
            )
        self.num_activities = num_activities

        self.d_model = d_model

        # Cardinality categoricals encoder prefix events
        self.cardinality_categoricals_pref = cardinality_categoricals_pref
        self.num_categoricals_pref = len(self.cardinality_categoricals_pref)

        # Number of numerical features encoder prefix events 
        self.num_numericals_pref = num_numericals_pref

        self.num_prefix_encoder_layers = num_prefix_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        # self.remaining_runtime_head = remaining_runtime_head
        self.layernorm_embeds = layernorm_embeds

        # Initialize positional encoding layer 
        self.positional_encoding = PositionalEncoding(d_model)

        # Initializing the categorical embeddings for the encoder inputs: 
        # Shared activity embeddings prefix and suffix! So only for the remaining ones you should do it. 
        self.embed_sz_categ_pref = [min(600, round(1.6 * n_cat**0.56)) for n_cat in self.cardinality_categoricals_pref[:-1]]
        self.activity_emb_size = min(600, round(1.6 * self.cardinality_categoricals_pref[-1]**0.56))

        # Initializing a separate embedding layer for each categorical prefix feature 
        # (Incrementing the cardinality with 1 to account for the padding idx of 0.)
        self.cat_embeds_pref = nn.ModuleList([nn.Embedding(num_embeddings=self.cardinality_categoricals_pref[i]+1, embedding_dim=self.embed_sz_categ_pref[i], padding_idx=0) for i in range(self.num_categoricals_pref-1)])
        self.act_emb = nn.Embedding(num_embeddings=num_activities-1, embedding_dim=self.activity_emb_size, padding_idx=0)


        # Dimensionality of initial encoder events after the prefix categoricals are fed to the dedicated entity embeddings and everything, including the numericals 
        # are concatenated
        self.dim_init_prefix = sum(self.embed_sz_categ_pref) + self.activity_emb_size + self.num_numericals_pref
        # Initial input embedding prefix events (encoder)
        self.input_embeddings_encoder = nn.Linear(self.dim_init_prefix, self.d_model)

        # Dimensionality initial decoder suffix event tokens. 
        # If the single seqeuential task to be trained is the 
        #   - activity suffix : only the decoded activities can be kept 
        #                       track of during AR decoding. 
        #   - timestamp suffix : only the decoded timestamps can be kept 
        #                        track of during AR decoding, and hence 
        #                        only the two time features can be 
        #                        updated. 
        if self.act_sufbool: 
            self.dim_init_suffix = self.activity_emb_size
        elif self.ts_sufbool: 
            self.dim_init_suffix = 2

        # Initial input embedding suffix events (decoder)
        self.input_embeddings_decoder = nn.Linear(self.dim_init_suffix, self.d_model)

        # Initializing the num_prefix_encoder_layers encoder layers 
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(self.num_prefix_encoder_layers)])
        # Initializing the num_decoder_layers decoder layers 
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(self.num_decoder_layers)])

        # Initializing the fully connected final prediction head on top 
        # of the decoder 
        if self.act_sufbool: 
            # Activity suffix prediction
            self.fc_final_out = nn.Linear(self.d_model, self.num_activities)
        elif self.ts_sufbool: 
            # Timestamp suffix prediction
            self.fc_final_out = nn.Linear(self.d_model, 1)
        
        
        if self.layernorm_embeds:
            self.norm_enc_embeds = nn.LayerNorm(self.d_model)
            self.norm_dec_embeds = nn.LayerNorm(self.d_model)

            
        self.dropout = nn.Dropout(self.dropout)
    


    # window_size : number of decoding steps during inference (model.eval())
    def forward(self, 
                inputs, 
                window_size=None, 
                mean_std_ttne=None, 
                mean_std_tsp=None, 
                mean_std_tss=None):
        """Processing a batch of inputs. The activity labels of the 
        prefix events are (and should) always be located at 
        inputs[self.num_categoricals_pref-1].

        Parameters
        ----------
        inputs : list of torch.Tensor
            Tensors already subset for the SST task at hand (e.g., via
            `SST_SuTraN.convert_tensordata.subset_data`). The tuple still
            includes the complete prefix stack plus every suffix-event helper
            tensor (activity indices and both time proxy features).
            The `SST_SuTraN.SuTraN_SST` forward pass
            then consumes only the suffix tokens it can update
            autoregressively (activity indices for
            `seq_task='activity_suffix'`, or the two time numerics for
            `seq_task='timestamp_suffix'`), ignoring the other suffix tensor
            because that head is not being generated. 
        window_size : None or int, optional
            The (shared) sequence length of the prefix and suffix inputs. 
            Only needed during inference (`model.eval()`). The default 
            value `None` can be retained during training. 
        mean_std_ttne : None or list of float, optional 
            List of two floats representing the mean and standard 
            deviation of the Time Till Next Event (TTNE) prediction 
            targets in seconds, computed over the training set instances 
            and used to standardize the TTNE labels of the training set, 
            validation set and test set. Needed for converting 
            timestamp predictions back to seconds and vice versa, only 
            during inference (`model.eval()`) when 
            `seq_task='timestamp_suffix'`. The default value `None` can 
            be retained during training, and even during inference in 
            case `seq_task='activity_suffix'`. 
        mean_std_tsp : None or list of float, optional 
            List of two floats representing the mean and standard 
            deviation of the Time Since Previous (TSP) event features of 
            the suffix event tokens, in seconds computed over the 
            training set instances and used to standardize the TSP values 
            of the training set, validation set and test set. 
            Only needed during inference (`model.eval()`) when 
            `seq_task='timestamp_suffix'`. The default value `None` can 
            be retained during training, and even during inference in 
            case `seq_task='activity_suffix'`.  
        mean_std_tss : None or list of float, optional 
            List of two floats representing the mean and standard 
            deviation of the Time Since Start (TSS) event features of 
            the suffix event tokens, in seconds computed over the 
            training set instances and used to standardize the TSS values 
            of the training set, validation set and test set. 
            Only needed during inference (`model.eval()`) when 
            `seq_task='timestamp_suffix'`. The default value `None` can 
            be retained during training, and even during inference in 
            case `seq_task='activity_suffix'`.  
        """
        # Tensor containing the numerical features of the prefix events. 
        num_ftrs_pref = inputs[self.num_categoricals_pref] # (batch_size, window_size, N)

        # Tensor containing the padding mask for the prefix events. 
        padding_mask_input = inputs[self.num_categoricals_pref+1] # (batch_size, window_size) = (B, W)

        # Just auxiliary index for understandability
        idx = self.num_categoricals_pref+2

        # Tensor containing the two timestamp-related numerical features 
        # of the suffix event tokens (TSS and TSP) 
        # NOTE: only needed for timestamp suffix prediction
        if self.ts_sufbool: 
            num_ftrs_suf = inputs[idx + 1] # (batch_size, window_size, 2)

        # Constructing categorical embeddings prefix (encoder)
        cat_emb_pref = self.cat_embeds_pref[0](inputs[0]) # (batch_size, window_size, embed_sz_categ[0])
        for i in range(1, self.num_categoricals_pref-1):
            cat_emb_help = self.cat_embeds_pref[i](inputs[i]) # (batch_size, window_size, embed_sz_categ[i])
            cat_emb_pref = torch.cat((cat_emb_pref, cat_emb_help), dim = -1) # (batch_size, window_size, sum(embed_sz_categ[:i+1]))
        act_emb_pref = self.act_emb(inputs[self.num_categoricals_pref-1])
        cat_emb_pref = torch.cat((cat_emb_pref, act_emb_pref), dim=-1)
        
        # Concatenate cat_emb with the numerical features to get initial vector representations prefix events. 
        x = torch.cat((cat_emb_pref, num_ftrs_pref), dim = -1) # (batch_size, window_size, sum(embed_sz_categ)+N)

        # Dropout over concatenated features: 
        x = self.dropout(x)

        # Initial embedding encoder (prefix events)
        x = self.positional_encoding(self.input_embeddings_encoder(x) * math.sqrt(self.d_model)) # (batch_size, window_size, d_model)
        if self.layernorm_embeds:
            x = self.norm_enc_embeds(x) # (batch_size, window_size, d_model)

        # Updating the prefix event embeddings with the encoder blocks 
        for enc_layer in self.encoder_layers:
            x = enc_layer(x, padding_mask_input)

        # ---------------------------

        if self.training: # Teacher forcing (for now)

            # Using the activity embedding layer shared with the encoder 
            if self.act_sufbool: 
                target_in = self.act_emb(inputs[idx]) # (batch_size, window_size, embed_sz_categ[0])
            
            elif self.ts_sufbool: 
                target_in =  num_ftrs_suf # (batch_size, window_size, 2)

            # Initial embeddings decoder suffix event tokens 
            # The positional encoding module applies dropout over the result 
            target_in = self.positional_encoding(self.input_embeddings_decoder(target_in) * math.sqrt(self.d_model)) # (batch_size, window_size, d_model)

            if self.layernorm_embeds:
                target_in = self.norm_dec_embeds(target_in) # (batch_size, window_size, d_model)

            # Activating the decoder
            dec_output = target_in
            for dec_layer in self.decoder_layers:
                dec_output = dec_layer(dec_output, x, padding_mask_input) # (batch_size, window_size)

            # Shape: 
            #   - (batch_size, window_size, self.num_activities) in case of activity suffix prediction
            #   - (batch_size, window_size, 1) in case of timestamp suffix prediction
            suffix_pred = self.fc_final_out(dec_output) 

            return suffix_pred 

        else: # Inference mode greedy decoding activities

            # Retrieving suffix activity integer vector `act_inputs`.
            #   `act_inputs` still contains the ground truth activity 
            #   labels (shifted by 1) for the entire suffixes. However, 
            #   at each decoding step `dec_step`, we will only predict 
            #   based on the shifted suffix generated up till that point, 
            #   and use those predictions to update the activity labels  
            #   for the subsequent decoding step. Finally, the look-ahead  
            #   mask ensures that the decoder cannot incorporate any 
            #   information regarding ground-truth activity labels in the 
            #   suffix.
            #   NOTE: the same holds for the two time features of the 
            #   suffix event tokens (`num_ftrs_suf`).
            if self.act_sufbool: 
                act_inputs = inputs[idx] # (B, W)
                batch_size = act_inputs.size(0) # B
                suffix_acts_decoded = torch.full(size=(batch_size, window_size), fill_value=0, dtype=torch.int64).to(device) # (B, W)

            elif self.ts_sufbool: 
                batch_size = num_ftrs_suf.size(0)
                suffix_ttne_preds = torch.full(size=(batch_size, window_size), fill_value=0, dtype=torch.float32).to(device) # (B, W)

            # Initializing zero filled tensors for storing the activity 
            # and timestamp predictions during decoding 
            # suffix_acts_decoded = torch.full(size=(batch_size, window_size), fill_value=0, dtype=torch.int64).to(device) # (B, W)
            # suffix_ttne_preds = torch.full(size=(batch_size, window_size), fill_value=0, dtype=torch.float32).to(device) # (B, W)

            for dec_step in range(0, window_size):

                if self.act_sufbool: 
                    target_in = self.act_emb(act_inputs) # (B, W, self.activity_emb_size)
                elif self.ts_sufbool: 
                    target_in = num_ftrs_suf # (B, W, 2)

                # Initial embeddings decoder suffix event tokens 
                target_in = self.positional_encoding(self.input_embeddings_decoder(target_in) * math.sqrt(self.d_model)) # (B, W, d_model)

                # Applying layernorm if specified 
                if self.layernorm_embeds:
                    target_in = self.norm_dec_embeds(target_in) # (B, W, d_model)

                # Activating the decoder
                dec_output = target_in
                for dec_layer in self.decoder_layers:
                    dec_output = dec_layer(dec_output, x, padding_mask_input) # (batch_size, window_size)

                if self.act_sufbool: 
                    act_logits = self.fc_final_out(dec_output) # (B, W, self.num_activities)
                    act_outputs = act_logits[:, dec_step, :] # (B, C)
                    # Decoding activity preditions (greedily)
                    #   "Masking padding token"
                    act_outputs[:, 0] = -1e9
                    #   Greedy selection 
                    act_selected = torch.argmax(act_outputs, dim=-1) # (batch_size,), torch.int64
                    #   Adding selected activity integers to suffix_acts_decoded
                    suffix_acts_decoded[:, dec_step] = act_selected

                    if dec_step < (window_size-1): 
                        # Deriving activity indices pertaining to the 
                        # selected activities for the derived next suffix 
                        # event to be fed to the decoder in the next decoding 
                        # step. 
                        act_suf_updates = act_selected.clone() # (batch_size, )

                        #   There is no artificially added END token present in the 
                        #   suffix activity representations, and hence there is no 
                        #   end token index in the suffix activity representations 
                        #   on index num_activities-1. Therefore, we clamp 
                        #   it on num_activities-2. Predictions for already finished 
                        #   instances will not be taken into account at the end. 
                        act_suf_updates = torch.clamp(act_suf_updates, max=self.num_activities-2) # (batch_size,) aka (B,)

                        # Updating `act_inputs` for suffix decoder for next decoding step 
                        act_inputs[:, dec_step+1] = act_suf_updates # (B, W)


                elif self.ts_sufbool: 
                    ttne_pred = self.fc_final_out(dec_output) # (B, W, 1)
                    ttne_outputs = ttne_pred[:, dec_step, 0] # (B, )
                    # Adding time pred as-is 
                    suffix_ttne_preds[:, dec_step] = ttne_outputs # (B, W)

                    if dec_step < (window_size-1): 
                        d = 1 
                        #   Converting predictions standardized TTNE 
                        #   back to original scale (seconds)
                        time_preds_seconds = ttne_outputs*mean_std_ttne[1] + mean_std_ttne[0] # (batch_size,)

                        #   Truncating at zero (no negatives allowed)
                        time_preds_seconds = torch.clamp(time_preds_seconds, min=0)

                        #   Converting standardized TSS feature current decoding 
                        #   step's suffix event token to original scale (seconds) 
                        tss_stand = num_ftrs_suf[:, dec_step, 0].clone() # (batch_size,)
                        tss_seconds = tss_stand*mean_std_tss[1] + mean_std_tss[0] # (batch_size,)

                        #   Clamping at zero again 
                        tss_seconds = torch.clamp(tss_seconds, min=0)

                        #   Updating tss in seconds next decoding step based on 
                        #   converted TTNE predictions 
                        tss_seconds_new = tss_seconds + time_preds_seconds # (batch_size,)

                        #   Converting back to preprocessed scale based on 
                        #   training mean and std
                        tss_stand_new = (tss_seconds_new - mean_std_tss[0]) / mean_std_tss[1] # (batch_size,)

                        #   TSP: time since previous event next decoding step 
                        #   is equal to the ttne in seconds, standardized with 
                        #   the training mean and std of the Suffix TSP feature 
                        tsp_stand_new = (time_preds_seconds - mean_std_tsp[0]) / mean_std_tsp[1] # (batch_size,)


                        #   Concatenating both 
                        new_suffix_timefeats = torch.cat((tss_stand_new.unsqueeze(-1), tsp_stand_new.unsqueeze(-1)), dim=-1) # (B, 2)
                        #   Updating next decoding step's time feature
                        #   tensor for the suffix event tokens 
                        num_ftrs_suf[:, dec_step+1, :] = new_suffix_timefeats # (B, W, 2)            

            if self.act_sufbool: 
                return suffix_acts_decoded 
            elif self.ts_sufbool: 
                return suffix_ttne_preds