"""
Module containing the parameter settings for the different 
MTO techniques used in the final experimental setup of the SuTraN+
paper. 
"""

######################################
# GradNorm Configurations
######################################
gradnorm_config_dict = {'lr_model' : 0.0002,
                        'lr_gradnorm' : 0.025,
                        'alpha' : 0.15,
                        'exponential_transformation' : False, 
                        'warmup_epoch' : True,
                        'theoretical_initial_loss' : False
                        }


######################################
# Uncertainty Weighting Configurations
######################################
uw_config_dict = {'lr_model' : 0.0002,
                  'init_logsigmas' : -1.5,
                  'softmax_normalization' : False
                   }


######################################
# UW+ Configurations
######################################
uw_plus_config_dict = {'lr_model' : 0.0002,
                       'init_logsigmas' : -1.5,
                       'softmax_normalization' : True
                       }


######################################
# PCGrad Configurations
######################################
# # Maybe this could just be hardcoded in the 
# # actual train eval call. Just like lr for the other techniques. 
# pcgrad_config_dict = {'lr_model' : 0.0002}

######################################
# NSST SuTraN baseline Configurations
######################################
# nsst_config_dict = {'lr' : 0.0002, 
#                    'final_embedding' : 'last'}
nsst_config_dict = {'final_embedding' : 'last'}
