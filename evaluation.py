# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:24:26 2023

@author: davarakis
"""

import tensorflow as tf
import keras_tuner as kt

import pandas as pd

import audio_processing as pr 
import run_models as rm


import numpy as np
import json
from sklearn.model_selection import train_test_split


# Currently model_name is : CNN_S_6_A or CNN_F_1_A
def evaluate_model(model_name):
    if (model_name == 'CNN_S_6_A'):
        features_filename = "gtzan/features-3sec"
        labels_filename = "gtzan/labels-3sec"
        
        features = rm.read_list_fromfile(features_filename)
        print('Length of read features is: ', len(features))
        
        labels = rm.read_list_fromfile(labels_filename)
        print('Length of read labels is: ', len(labels))
        
        features_train, labels_train, features_val, labels_val, features_test, labels_test = pr.prepareStatisticsDataset(features, labels)
        features_test = features_test.reshape(features_test.shape[0], 83, 6, 1)
    else:
        filename ="gtzan/second-all-3secs.json"

        with open(filename) as f:
            data = json.load(f)
            f.close()
        # turn data into numpy arrays
        inputs = np.array(data["mfcc"])
        labels = np.array(data["labels"])
        inputs_training, inputs_test, labels_training, labels_test = train_test_split(inputs, labels, test_size=0.2, shuffle=True, random_state=2023)
        #inputs_train, inputs_val, labels_train, labels_val = train_test_split(inputs_training, labels_training, test_size=0.2, shuffle=True, random_state=2023)
        #print("Length of inputs_train:", inputs_train.shape)
        #print("Length of labels_train:", labels_train.shape)
        #print("Length of inputs_val:", inputs_val.shape)
        #print("Length of labels_val:", labels_val.shape)

    tuned_model_name = "savedmodels/"+ model_name+"_tunned.h5"
    new_model = tf.keras.models.load_model(tuned_model_name)

    # Show the model architecture
    model_summary = new_model.summary()
    if (model_name == 'CNN_S_6_A'):
        loss, acc = new_model.evaluate(features_test, labels_test, verbose=2)
    else:
        loss, acc = new_model.evaluate(inputs_test, labels_test, verbose=2)

    
    print("Loss: ", loss)
    
    print("Accuracy: {:5.2f}%".format(100 * acc))
    
    return model_summary, loss, acc

# For testing
#loss, acc = evaluate_model("CNN_S_6_A")
