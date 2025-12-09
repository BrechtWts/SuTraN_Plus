"""
Code of the Non-Sequential, Single-Task (NSST) (encoder-only) version of
SuTraN for the two non-sequential prediction tasks, being Remaining
Runtime (RRT) prediction and Outcome prediction (both Binary Outcome (BO)
and Multi-Class Outcome (MCO)).
"""


import torch 
import torch.nn as nn
import math

from SuTraN.transformer_prefix_encoder import EncoderLayer

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


class SuTraN_NSST(nn.Module):
    def __init__(self, 
                 num_activities, 
                 d_model, 
                 cardinality_categoricals_pref, 
                 num_numericals_pref, 
                 scalar_task, 
                 num_outclasses=None, 
                 num_prefix_encoder_layers=4, 
                 num_heads=8, 
                 d_ff = 128, 
                 dropout = 0.2,
                 final_embedding='last', 
                 layernorm_embeds=True
                 ):
        """Initialize an instance of the Non-Sequential, Single-Task 
        (NSST), encoder-only, version of SuTraN. To be trained for 
        one of the three Non-Sequential prediction targets, specified by 
        the `scalar_task` argument. 

        #. Remaining Runtime (RRT) prediction
        #. Binary Outcome (BO) prediction
        #. Multi-Class Outcome (MCO) prediction

        Parameters
        ----------
        num_activities : int
            Number of distinct activities present in the event log. 
            This does include the end token and padding token 
            used for the activity labels. For the categorical activity 
            label features in the prefix and suffix, no END token is 
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
            Number of numerical features of the prefix events. 
        scalar_task : {'remaining_runtime', 'binary_outcome', 'multiclass_outcome'}
            The scalar prediction task trained and evaluated for. Either 
            `'remaining_runtime'` `'binary_outcome'` or 
            `'multiclass_outcome'`.
        num_outclasses : int or None, optional
            The number of outcome classes in case 
            `scalar_task='multiclass_outcome'`. By default `None`. 
        num_prefix_encoder_layers : int, optional
            The number of prefix encoder blocks stacked on top of each 
            other, by default 4.
        num_heads : int, optional
            Number of attention heads for the Multi-Head Attention 
            sublayers in both the encoder and decoder blocks, by default 
            8.
        d_ff : int, optional
            The dimension of the hidden layer of the point-wise feed 
            forward sublayers in the transformer blocks , by default 128.
        dropout : float, optional
            Dropout rate during training. By default 0.2. 
        final_embedding : {'CLS', 'last'}
            Indicates which encoder embedding feeds the scalar head:
            `'CLS'` consumes an explicit CLS token (if prepended in the
            data), whereas `'last'` uses the final non-padded prefix
            event. This switch was kept for early ablation purposes;
            preliminary trials showed `'last'` to be consistently stronger,
            so all paper results fix this parameter to `'last'`. Default
            is `'last'`.
        layernorm_embeds : bool, optional
            Whether or not Layer Normalization is applied over the 
            initial embeddings of the encoder and decoder. True by 
            default.
        """
        super(SuTraN_NSST, self).__init__()
        self.num_outclasses = num_outclasses
        self.outcome_bool = (scalar_task=='binary_outcome') or (scalar_task=='multiclass_outcome')
        # binary out bool
        self.bin_outbool = (scalar_task=='binary_outcome')
        # multiclass out bool
        self.multic_outbool = (scalar_task=='multiclass_outcome')
        if self.multic_outbool: 
            if self.num_outclasses is None: 
                raise ValueError(
                    "When `scalar_task='multiclass_outcome'`, "
                    "'num_outclasses' should be given an integer argument."
                )

        self.num_activities = num_activities
        self.d_model = d_model
        self.cardinality_categoricals_pref = cardinality_categoricals_pref
        self.num_categoricals_pref = len(self.cardinality_categoricals_pref)
        self.num_numericals_pref = num_numericals_pref
        self.num_prefix_encoder_layers = num_prefix_encoder_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.final_embedding = final_embedding
        self.layernorm_embeds = layernorm_embeds

        # Initialize positional encoding layer 
        self.positional_encoding = PositionalEncoding(d_model)

        if self.final_embedding=='CLS': 
            # Initializing the categorical embeddings for the encoder inputs:  
            # Incrementing the original cardinality with 1, to account for the added CLS token. 
            self.embed_sz_categ_pref = [min(600, round(1.6 * (n_cat+1)**0.56)) for n_cat in self.cardinality_categoricals_pref[:-1]]
            self.activity_emb_size = min(600, round(1.6 * (self.cardinality_categoricals_pref[-1]+1)**0.56))
            # Initializing a separate embedding layer for each categorical prefix feature 
            # (Incrementing the cardinality with 2 to account for the padding idx of 0, and 
            # the added CLS token)
            self.cat_embeds_pref = nn.ModuleList([nn.Embedding(num_embeddings=self.cardinality_categoricals_pref[i]+2, embedding_dim=self.embed_sz_categ_pref[i], padding_idx=0) for i in range(self.num_categoricals_pref-1)])
            self.act_emb = nn.Embedding(num_embeddings=num_activities, embedding_dim=self.activity_emb_size, padding_idx=0)

        elif self.final_embedding=='last':
            # Initializing the categorical embeddings for the encoder inputs:  
            # No incrementation with one should be done to determine the embedding 
            # dimensionalities, since the added padding token at index 0 does not carry 
            # any meaning. The original cardinality (excluding the padding token) 
            # can be used. 
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

        # Initializing the num_prefix_encoder_layers encoder layers 
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(self.num_prefix_encoder_layers)])

        # Initialize remaining runtime prediction head 
        if not self.outcome_bool: # RRT prediction
            self.fc_prediction_head = nn.Linear(self.d_model, 1)
        
        elif self.bin_outbool: 
            self.fc_prediction_head = nn.Linear(self.d_model, 1)
            self.sigmoid_out = nn.Sigmoid()
        
        elif self.multic_outbool: 
            self.fc_prediction_head = nn.Linear(self.d_model, self.num_outclasses)

        # Initialize layernorm on top of initial prefix event embeddings
        if self.layernorm_embeds:
            self.norm_enc_embeds = nn.LayerNorm(self.d_model)

        # Init dropout 
        self.dropout = nn.Dropout(self.dropout)
    
    def forward(self, 
                inputs):
        """Processing a batch of instances. 

        Parameters
        ----------
        inputs : list of torch.Tensor
            List of tensors containing the various components of the
            inputs. Before constructing this list, convert the SuTraN(+)
            tensors with `NSST_SuTraN.convert_tensordata.subset_data` so
            only the scalar target and its required inputs remain. When
            `final_embedding='CLS'`, additionally run
            `NSST_SuTraN.convert_tensordata.add_prefix_CLStoken` so a CLS
            token is prepended to each prefix sequence.
        Returns
        -------
        torch.Tensor
            Scalar predictions for the requested task: `(batch_size, 1)`
            for remaining time or binary outcome, and
            `(batch_size, num_outclasses)` for multiclass outcome.
        """
        # Tensor containing the numerical features of the prefix events. 
        num_ftrs_pref = inputs[self.num_categoricals_pref] # (batch_size, window_size, N)

        # Tensor containing the padding mask for the prefix events. 
        # padding_mask_input = inputs[(self.num_categoricals_pref-1)+2] # (batch_size, window_size) = (B, W)
        padding_mask_input = inputs[-1] # (batch_size, window_size) = (B, W)



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
            x = enc_layer(x, padding_mask_input) # (batch_size, window_size, d_model)


        if self.final_embedding=='CLS': 
            # In this case, the embedding of the CLS token, which is 
            # appended to the beginning of the prefix event sequences, 
            # are fed to the final prediction head. 
            # NOTE: in this case, the middle dimension of the embeddings 
            # is window_size+1
            ultimate_embeddings = x[:, 0, :] # shape (batch_size, d_model)
            

        
        elif self.final_embedding=='last':
            non_padded_mask = ~padding_mask_input  # False -> True for actual tokens

            # Find the last non-padded index for each sequence
            # by locating the last True in reverse order
            last_indices = non_padded_mask.int().cumsum(dim=1).argmax(dim=1)

            # Extract the embeddings of the last actual tokens
            ultimate_embeddings = x[torch.arange(x.size(0)), last_indices]  # shape (batch_size, d_model)


        # Use the extracted embedding (CLS or last) to predict non-sequential target 
        # shape (batch_size, 1) in case of RRT or BO, shape (batch_size, num_outclasses) 
        # in case of MCO. 
        pred = self.fc_prediction_head(ultimate_embeddings)  

        if self.bin_outbool: 
            pred = self.sigmoid_out(pred) # (batch_size, 1)

        return pred


