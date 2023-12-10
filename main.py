"""
Created on Tue Apr 11 15:24:26 2023

@author: davarakis
"""
from __future__ import print_function
import os

import json
import argparse

import numpy as np
import pandas as pd
 
import utils.params as par
from utils.plotting import plot_training

import run_models as rm 

# Reads arguments to get :
# 1. the model name that will be trained
# 2. the name of the log file that will be produced
# 3. indication of whether dataset will be preprocessed
def main():
    #start_time = time.strftime("%d%m%y_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name", 
        type=str, 
        help="Pass name of model as defined in hparams.yaml."
        )
    parser.add_argument(
        "log_filename",
        type=str, 
        help="Pass name of log file"
        )
    parser.add_argument(
        "--write_data",
        required = False,
        default=False,
                help="Set to true to write_data."
        )
    parser.add_argument(
        "--read_features",
        required = False,
        default=False,
                help="Set to true to read preprocessed data."
        )
    args = parser.parse_args()

    # Get the parameters for building and training the model   
    # Parameters are defined in params.xls file 
    # and are processed by utils/params.py script
    params = par.get_params(args.model_name)
    print("PARAMS are :", params)
    
    #model_name = params["model_name"].split("_")
    #print("model_name", model_name)
    
    if not os.path.exists("gtzan"): os.makedirs("gtzan")
    
    # Decide the type of the workflow and run the relevant function to
    # process the data and train the model
    if params["model_type"] == "s":
        history, features_test, labels_test = rm.run_statistical_model(args.read_features, args.write_data, params)
    else:
        history, features_test, labels_test = rm.run_audiofeatures_model(args.read_features, args.write_data, params)

    #print("HISTORY KEYS:", history.history.keys())

    # Get performance results to preppare the log file
    training_accuracy = history.history['acc']
    validation_accuracy = history.history['val_acc']
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    # Print the loss and accuracy plots
    fig = plot_training(training_loss, training_accuracy,validation_loss, validation_accuracy)
    fig.savefig(os.path.join("figs", "{}_training_vis".format(args.model_name)))

    # Some log information to help you keep track of your model information. 
    #print("PARAMS:", params["lr"])
    logs ={
            "model": args.model_name,
            "train_losses": training_loss,
            "train_accs": training_accuracy,
            "val_losses": validation_loss,
            "val_accs": validation_accuracy,
            "best_val_epoch": int(np.argmax(validation_accuracy)+1),
            "lr": params["lr"],
            "num_epochs": int(params["num_epochs"]),
            "batch_size":int(params["batch_size"])
        }

    log_filename = args.log_filename
    #print(" log_filename:" , log_filename)
    with open(os.path.join(params["log_dir"],log_filename), 'w') as f:
        json.dump(logs, f)


if __name__ == '__main__':
   main()
