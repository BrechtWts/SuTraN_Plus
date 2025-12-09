"""
Module containing the event log (dataset) properties 
for the event logs used in the final experimental setup of the SuTraN+
paper.
"""

# The log names are:
log_name_list = ['BPIC_19', 'BPIC_17_DR', 'BPIC_17']


# Defining the log-specific parameters in dictionaries 
#   median_caselen : the median case length of the log
median_caselen_dict = {'BPIC_19': 5, 
                       'BPIC_17_DR': 21, 
                       'BPIC_17': 34}

#   outcome_bool 
outcome_bools_dict = {'BPIC_19': False,
                      'BPIC_17_DR': True,
                      'BPIC_17': True}

#   out_mask
out_masks_dict = {'BPIC_19' : False, 
                  'BPIC_17_DR' : True, 
                  'BPIC_17' : True}

#   out_type
out_types_dict = {'BPIC_19' : None, 
                  'BPIC_17_DR' : 'multiclass_outcome', 
                  'BPIC_17' : 'multiclass_outcome'}
#   num_outclasses
num_outclasses_dict = {'BPIC_19' : None, 
                       'BPIC_17_DR' : 3, 
                       'BPIC_17' : 3}
