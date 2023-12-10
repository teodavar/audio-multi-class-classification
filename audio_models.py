"""
Created on Tue Apr 11 15:24:26 2023

@author: davarakis
"""

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras
import keras.utils
from keras import utils as np_utils
import numpy as np
import os

# Define the audio models that will be used to run the experiments for
# Workflow 1 and 2

# Each model takes from its params variable the following information:
# learning rate, number of epochs, batch_size

# For all models:
# Default optimizer : RMSprop
# Early Stopping is activated
# Loss function: sparse categorical cross entropy
# evaluation measure: Accuracy

output_shape = 10

# Workflow 1 - MLP model
def mlp_s(features_train, labels_train, features_val, labels_val, params):
    input_shape = len(features_train)
    learning_rate = params["lr"]
    num_epochs = params["num_epochs"]
    batch_size = params["batch_size"]
    # Let's design the model architecture.
    #model_name = params["model_name"].split("_")

    if (params["model_id"] == "1"):
        model = tf.keras.models.Sequential([
            #tf.keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
            tf.keras.layers.InputLayer(input_shape=(498), name="feature"),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'), 
            tf.keras.layers.Dense(64, activation='relu'), 
            tf.keras.layers.Dense(output_shape, activation='softmax', name="predictions")
        ])
    model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        # Loss function to minimize
        loss=keras.losses.SparseCategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=['acc'],
             )

    model.summary()
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    history = model.fit(x=features_train.tolist(),y=labels_train.tolist(),
                            verbose=1,validation_data=(features_val.tolist() , labels_val.tolist()), 
                            epochs=num_epochs, batch_size=batch_size, callbacks=[callback])
    
    save_filename = params["savemodel_dir"]+"/"+ params["model_name"]+".h5" 
    model.save(save_filename)      
    return history

# Workflow 2 - MLP model
def mlp_f(features_train, labels_train, features_val, labels_val, params):
    learning_rate = params["lr"]
    num_epochs = params["num_epochs"]
    batch_size = params["batch_size"]
    input_shape = (features_train.shape[1], features_train.shape[2])
    print("input_shape: ", input_shape)
    
    # Let's design the model architecture.
    #model_name = params["model_name"].split("_")
    if (params["model_id"] == "1"):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(input_shape)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'), 
            tf.keras.layers.Dense(64, activation='relu'), 
            tf.keras.layers.Dense((output_shape), activation='softmax')
        ])

    model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        # Loss function to minimize
        loss=keras.losses.SparseCategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=['acc']
             )

    model.summary()
    
    #Training the model.
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    history = model.fit(x=features_train,y=labels_train,
                            verbose=1,validation_data=(features_val , labels_val), 
                            epochs=num_epochs, batch_size=batch_size, callbacks=[callback])      
    
    save_filename = params["savemodel_dir"]+"/"+ params["model_name"]+".h5" 
    model.save(save_filename)      
    return history

# Workflow 2 - CNN models
# 3 CNN models are defined 
def cnn_f(features_train, labels_train, features_val, labels_val, params):
   
    learning_rate = params["lr"]
    num_epochs = params["num_epochs"]
    batch_size = params["batch_size"]
    
    features_train = features_train[..., np.newaxis]
    labels_train = labels_train[..., np.newaxis]
    features_val = features_val[..., np.newaxis]
    labels_val = labels_val[..., np.newaxis]
   
    #input_shape = (Xtrain.shape[1], Xtrain.shape[2], 1)
    input_shape = (features_train.shape[1], features_train.shape[2], features_train.shape[3])
    
    #model_name = params["model_name"].split("_")
    if (params["model_id"] == "1"):
    # 1st CNN MOdel = 62%
        model = tf.keras.models.Sequential([
                
                tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
                tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same'),
                tf.keras.layers.BatchNormalization(),
                
                tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same'),
                tf.keras.layers.BatchNormalization(),
                
                tf.keras.layers.Conv2D(32, (2,2), activation='relu'),
                tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'), 
                tf.keras.layers.Dense((output_shape), activation='softmax')
            ])
    elif (params["model_id"] == "2"):
        model = tf.keras.models.Sequential([
                
                tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),       
                tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
                tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
                tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2,2), strides=(2,2), padding='same'),
                tf.keras.layers.BatchNormalization(),
        
                tf.keras.layers.Conv2D(8, (2,2), activation='relu'),
                tf.keras.layers.Conv2D(8, (2,2), activation='relu'),
                tf.keras.layers.MaxPooling2D((2,2), strides=(2,2), padding='same'),
                tf.keras.layers.BatchNormalization(),
            
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'), 
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'), 
            
                tf.keras.layers.Dense((output_shape), activation='softmax')
            ])
    elif (params["model_id"] == "3"):
        model = tf.keras.models.Sequential([
                
                tf.keras.layers.Conv2D(128, (2,2), activation='relu', input_shape=input_shape),   
                tf.keras.layers.Conv2D(64, (2,2), activation='relu'),
        
                tf.keras.layers.BatchNormalization(),
                
                tf.keras.layers.Conv2D(32, (2,2), activation='relu'),
                tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
            
                tf.keras.layers.Conv2D(32, (2,2), activation='relu'),
                tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'), 
                tf.keras.layers.Dense((output_shape), activation='softmax')
            ])

    model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        # Loss function to minimize
        loss=keras.losses.SparseCategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=['acc'],
             )

    model.summary()
    
    #Training the model.
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    history = model.fit(x=features_train,y=labels_train,
                            verbose=1,validation_data=(features_val , labels_val), 
                            epochs=num_epochs, batch_size=batch_size, callbacks=[callback])      
    
    save_filename = params["savemodel_dir"]+"/"+ params["model_name"]+".h5" 
    model.save(save_filename)      
    return history

# Workflow 1 - CNN models
# 8 CNN models are defined
def cnn_s(features_train, labels_train, features_val, labels_val, params):
   
    learning_rate = params["lr"]
    num_epochs = params["num_epochs"]
    batch_size = params["batch_size"]
    
    # CNN model expects 3D input shape.
    # 83*6 = 498
    features_train = features_train.reshape(features_train.shape[0], 83, 6, 1)
    features_val = features_val.reshape(features_val.shape[0], 83, 6, 1)
   
    #input_shape = (Xtrain.shape[1], Xtrain.shape[2], 1)
    input_shape = (features_train.shape[1], features_train.shape[2], features_train.shape[3])
    
    #model_name = params["model_name"].split("_")
    if (params["model_id"] == "1"):
    # 1st CNN MOdel = 62%
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(128, (2,2), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same'),
            tf.keras.layers.BatchNormalization(),
                    
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'), 
            tf.keras.layers.Dense(output_shape, activation='softmax', name="predictions")
            
        ])
    elif (params["model_id"] == "2"):
    # 2nd CNN MOdel = 57%
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D((3,3), strides=(2,1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D((3,3), strides=(2,1), padding='same'),
            tf.keras.layers.BatchNormalization(),
    
            tf.keras.layers.Conv2D(32, (2,2), activation='relu'),
            tf.keras.layers.MaxPooling2D((3,3), strides=(2,1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'), 
            tf.keras.layers.Dense(output_shape, activation='softmax')
            
        ])
    elif (params["model_id"] == "3"):
        # 3rd CNN: Accuracy = 57%
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (2,2), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D((3,3), strides=(2,1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Conv2D(32, (2,2), activation='relu'),
            tf.keras.layers.MaxPooling2D((3,3), strides=(2,1), padding='same'),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(32, (2,2), activation='relu'),
            tf.keras.layers.MaxPooling2D((3,3), strides=(2,1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'), 
            tf.keras.layers.Dense(output_shape, activation='softmax')
            
        ])
    elif (params["model_id"] == "4"):        
        # 4o CNN: Accuracy = 66%
        model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(32, (2,2), activation='relu', input_shape=input_shape),
                tf.keras.layers.MaxPooling2D((3,3), strides=(2,1), padding='same'),
                tf.keras.layers.BatchNormalization(),
                
                tf.keras.layers.Conv2D(32, (2,2), activation='relu'),
                tf.keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='same'),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.Conv2D(32, (2,2), activation='relu'),
                tf.keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'), 
                tf.keras.layers.Dense(output_shape, activation='softmax')
                            
        ])
    elif (params["model_id"] == "5"):                   
            # 5o CNN: Accuracy = 54%
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(32, (2,2), activation='relu', input_shape=input_shape),
                tf.keras.layers.MaxPooling2D((3,3), strides=(2,1), padding='same'),
                tf.keras.layers.BatchNormalization(),
                
                tf.keras.layers.Conv2D(32, (2,2), activation='relu'),
                tf.keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='same'),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.Conv2D(32, (2,2), activation='relu'),
                tf.keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                
                tf.keras.layers.Conv2D(32, (2,2), activation='relu'),
                tf.keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'), 
                tf.keras.layers.Dense(output_shape, activation='softmax')
            ])
    elif (params["model_id"] == "6"):              
            # CNN Model: 66.8%
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(40, (2,2), activation='relu', input_shape=input_shape),
                tf.keras.layers.MaxPooling2D((3,3), strides=(2,1), padding='same'),
                tf.keras.layers.BatchNormalization(),
                
                tf.keras.layers.Conv2D(40, (2,2), activation='relu'),
                tf.keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='same'),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.Conv2D(40, (2,2), activation='relu'),
                tf.keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
            
                
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'), 
                tf.keras.layers.Dense(output_shape, activation='softmax')   
            ])
    elif (params["model_id"] == "7"):  
            # CNN Model = 63.3#
            # epochs = 500
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(64, (2,2), activation='relu', input_shape=input_shape),
                tf.keras.layers.MaxPooling2D((3,3), strides=(2,1), padding='same'),
                tf.keras.layers.BatchNormalization(),
                
                tf.keras.layers.Conv2D(64, (2,2), activation='relu'),
                tf.keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='same'),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.Conv2D(64, (2,2), activation='relu'),
                tf.keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
            
                
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'), 
                tf.keras.layers.Dense(output_shape, activation='softmax')                
            ])
    elif (params["model_id"] == "8"):           
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(64, (2,2), activation='relu', input_shape=input_shape),
                tf.keras.layers.MaxPooling2D((3,3), strides=(2,1), padding='same'),
                tf.keras.layers.BatchNormalization(),
                
                tf.keras.layers.Conv2D(64, (2,2), activation='relu'),
                tf.keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='same'),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.Conv2D(64, (2,2), activation='relu'),
                tf.keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
            
                
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'), 
                tf.keras.layers.Dense(output_shape, activation='softmax')
                
            ])   
            
    model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
            # Loss function to minimize
            loss=keras.losses.SparseCategoricalCrossentropy(),
            # List of metrics to monitor
            metrics=['acc'],
                 )

    model.summary()

        #callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    history = model.fit(x=features_train.tolist(),y=labels_train.tolist(),
                            verbose=1,validation_data=(features_val.tolist() , labels_val.tolist()), 
                            epochs=num_epochs, batch_size=batch_size)
        
    save_filename = params["savemodel_dir"]+"/"+ params["model_name"]+".h5" 
    model.save(save_filename)      
    return history

# Workflow 1 - LSTM model
def lstm_s(features_train, labels_train, features_val, labels_val, params):
    
    learning_rate = params["lr"]
    num_epochs = params["num_epochs"]
    batch_size = params["batch_size"]
   
    features_train = features_train.reshape(features_train.shape[0], 83, 6)
    features_val = features_val.reshape(features_val.shape[0], 83, 6)
       
    #input_shape = (Xtrain.shape[1], Xtrain.shape[2], 1)
    input_shape = (features_train.shape[1], features_train.shape[2])
    
    #model_name = params["model_name"].split("_")
    if (params["model_id"] == "1"): 
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(64, input_shape= input_shape, return_sequences=True),
            tf.keras.layers.LSTM(64),
            
            tf.keras.layers.Dense(64, activation='relu'), 
            #tf.keras.layers.Dropout(0.3),
    
            tf.keras.layers.Dense(output_shape, activation='softmax')
        ])
    
    model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
            # Loss function to minimize
            loss=keras.losses.SparseCategoricalCrossentropy(),
            # List of metrics to monitor
            metrics=['acc'],
                 )

    model.summary()

        #callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    history = model.fit(x=features_train.tolist(),y=labels_train.tolist(),
                            verbose=1,validation_data=(features_val.tolist() , labels_val.tolist()), 
                            epochs=num_epochs, batch_size=batch_size)
    
    save_filename = params["savemodel_dir"]+"/"+ params["model_name"]+".h5" 
    model.save(save_filename)      
    return history

# Workflow 2 - LSTM model
def lstm_f(features_train, labels_train, features_val, labels_val, params):
    
    learning_rate = params["lr"]
    num_epochs = params["num_epochs"]
    batch_size = params["batch_size"]
   
    input_shape = (features_train.shape[1], features_train.shape[2])
    
    #model_name = params["model_name"].split("_")
    if (params["model_id"] == "1"): 
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(64, input_shape= input_shape, return_sequences=True),
            tf.keras.layers.LSTM(64),
            
            tf.keras.layers.Dense(64, activation='relu'), 
            #tf.keras.layers.Dropout(0.3),
    
            tf.keras.layers.Dense(output_shape, activation='softmax')
        ])
    
    model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
            # Loss function to minimize
            loss=keras.losses.SparseCategoricalCrossentropy(),
            # List of metrics to monitor
            metrics=['acc'],
                 )

    model.summary()

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    history = model.fit(x=features_train,y=labels_train,
                            verbose=1,validation_data=(features_val , labels_val), 
                            epochs=num_epochs, batch_size=batch_size, callbacks=[callback])  
    
    save_filename = params["savemodel_dir"]+"/"+ params["model_name"]+".h5" 
    model.save(save_filename)      
    return history