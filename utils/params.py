"""
Created on Tue Apr 11 15:24:26 2023

@author: davarakis
"""
import pandas as pd

# Training parameters for all models build for Workflow 1 and Workflow 2
# are defined in params.xls file

# Read params.xls file and prepare a params dictionary that
# will hold the training parameters for each model (amongst other)
def get_params(model_name):
    params_df = pd.read_excel('params.xlsx')
    
    line = params_df.loc[params_df['model_name'] == model_name]
    
    row = params_df.index[params_df['model_name'] ==  model_name]
    row = row[0]
    
    sm_model_name = model_name.lower()
    model_name_parts = sm_model_name.split("_")
        
    params ={
      "model_name": sm_model_name,
      "mdname": model_name_parts[0] + "_" + model_name_parts[1],
      "data_dir": "gtzan",
      "log_dir": 'logs/'+sm_model_name,
      "num_epochs": line.at[row,'num_epochs'],
      "batch_size": line.at[row,'batch_size'],
      "lr": line.at[row,'lr'],
      "model_type": model_name_parts[1],
      "model_id": model_name_parts[2],
      "augment_data": model_name_parts[3],
      #"checkpoint_dir": 'checkpoints/'+sm_model_name+'/cp.ckpt',
      "savemodel_dir": 'savedmodels/'
    }
    
    return params

    


