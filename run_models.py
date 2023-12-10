"""
Created on Tue Apr 11 15:24:26 2023

@author: davarakis
"""
from __future__ import print_function
import os

import json


import numpy as np
from sklearn.model_selection import train_test_split


import audio_models
import audio_processing as pr 
import pickle

# Write data to a file
def write_list_tofile(a_list, filename):
    # store list in binary file so 'wb' mode
    with open(filename, 'wb') as fp:
        pickle.dump(a_list, fp)
        print('Done writing list into a binary file')

# Read data from a file
def read_list_fromfile(filename):
    # for reading also binary mode is important
    with open(filename, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list

# Workflow 1 - Prepare the data, Build a model and Train it
def run_statistical_model(read_features, write_data, params):
    # Decide whether to run on original data or on increased dataset
    if (params["augment_data"] == "n"):
        features_filename = "features"
        labels_filename = "labels"
    else:
        features_filename = "features-3sec"
        labels_filename = "labels-3sec"
    # Read preprocessed data (features + labels) if specified in command line arguments. 
    if read_features:       
        filename = os.path.join(params["data_dir"], features_filename)
        features = read_list_fromfile(filename)
        print('Length of read features is: ', len(features))
        
        filename = os.path.join(params["data_dir"], labels_filename)
        labels = read_list_fromfile(filename)
        print('Length of read labels is: ', len(labels))
    else:
        # Read & Process Data
        if (params["augment_data"] == "n"):
            features, labels = pr.calculateStatisticsFeatures()
        else:
            features, labels = pr.calculateStatisticsFeatures_3sec()
        print(" LEN OF FEATURES: ", len(features))
        
           
        # Write data if specified in command line arguments. 
        if write_data:       
            filename = os.path.join(params["data_dir"], features_filename)
            write_list_tofile(features, filename)
            filename = os.path.join(params["data_dir"], labels_filename)
            write_list_tofile(labels, filename)
    
    # Split data to training, evaluation and test dataset
    features_train, labels_train, features_val, labels_val, features_test, labels_test = pr.prepareStatisticsDataset(features, labels)

    # Create the necessary folder (if needed)
    if not os.path.exists(params["log_dir"]): os.makedirs(params["log_dir"])
    #if not os.path.exists(params["checkpoint_dir"]): os.makedirs(params["checkpoint_dir"])
    if not os.path.exists(params["savemodel_dir"]): os.makedirs(params["savemodel_dir"])
    if not os.path.exists("figs"): os.makedirs("figs")

    # Load & Train model that has been chosen via the command line arguments.
    # dynamic call of a function
    #model_name = params["model_name"].split("_")
    #mdname = model_name[0]+"_"+model_name[1]
    nn_model = getattr(audio_models, params["mdname"])

    history = nn_model(features_train, labels_train, features_val, labels_val, params)
    return history, features_test, labels_test

# Workflow 2 - Prepare the data, Build a model and Train it
def run_audiofeatures_model(read_features, write_data, params):
    # Decide whether to run on original dataset or on increased dataset
    if (params["augment_data"] == "n"):
        filename = os.path.join(params["data_dir"], 'second-all-10secs.json')
    else:
        filename = os.path.join(params["data_dir"], 'second-all-3secs.json')
    # Read preprocessed data (features + labels) if specified in command line arguments. 
    if read_features:    
        with open(filename) as f:
            data = json.load(f)
            f.close()
        # turn data into numpy arrays
        inputs = np.array(data["mfcc"])
        labels = np.array(data["labels"])

        print('Length of read features is: ', inputs.shape)
        print('Length of read labels is: ', labels.shape)
    else:
        # Read & Process Data
        if (params["augment_data"] == "n"):
            audio_data = pr.calculateAudioFeatures()
        else:
            audio_data = pr.calculateAudioFeatures_3sec()
        
        inputs = np.array(audio_data["mfcc"])
        labels = np.array(audio_data["labels"])
        print('Length of read features is: ', inputs.shape)
        print('Length of read labels is: ', labels.shape)
                 
        # Write data if specified in command line arguments. 
        if write_data:       
            with open(filename, 'w') as fp:
                json.dump(audio_data, fp)
                fp.close()
    
    # Split data to training, evaluation and test dataset

    inputs_training, inputs_test, labels_training, labels_test = train_test_split(inputs, labels, test_size=0.2, shuffle=True, random_state=2023)
    inputs_train, inputs_val, labels_train, labels_val = train_test_split(inputs_training, labels_training, test_size=0.2, shuffle=True, random_state=2023)
    print("Length of inputs_train:", inputs_train.shape)
    print("Length of labels_train:", labels_train.shape)
    print("Length of inputs_val:", inputs_val.shape)
    print("Length of labels_val:", labels_val.shape)

    # Create the neccessary folder (if needed)
    if not os.path.exists(params["log_dir"]): os.makedirs(params["log_dir"])
    #if not os.path.exists(params["checkpoint_dir"]): os.makedirs(params["checkpoint_dir"])
    if not os.path.exists(params["savemodel_dir"]): os.makedirs(params["savemodel_dir"])
    if not os.path.exists("figs"): os.makedirs("figs")

    # Load & Train model that has been chosen via the command line arguments.
    # dynamic call of a function
    #model_name = params["model_name"].split("_")
    #mdname = model_name[0]+"_"+model_name[1]
    nn_model = getattr(audio_models, params["mdname"])
    history = nn_model(inputs_train, labels_train, inputs_val, labels_val, params)
    
    
    return history, inputs_test, labels_test
