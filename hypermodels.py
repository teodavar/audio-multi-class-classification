"""
Created on Tue Apr 11 15:24:26 2023

@author: davarakis
"""
import tensorflow as tf
import keras_tuner as kt

from tensorflow import keras

# Defined the HyperModels to be fined tuned

# For each hypermodel the following hyperparameters search space is defined:
# Leaning rate: values=[1e-2, 1e-3, 1e-4]
# Optimizer: values=['sgd', 'rmsprop', 'adam']
# Batch size : [16, 32,64]

class CNN_S_7_A_HyperModel(kt.HyperModel):
    
    def build(self, hp):
        input_shape = (83, 6, 1)
        output_shape = 10
        
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
        
        hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        # Select optimizer    
        # Refer to https://stackoverflow.com/questions/67286051/how-can-i-tune-the-optimization-function-with-keras-tuner
        hp_optimizer = hp.Choice('optimizer', values=['sgd', 'rmsprop', 'adam'])

        if hp_optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=hp_learning_rate)
        elif hp_optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=hp_learning_rate)
        elif hp_optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
    
        model.compile(optimizer=optimizer,
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [16, 32,64]),
            **kwargs,
        )


class CNN_S_6_A_HyperModel(kt.HyperModel):
    
    def build(self, hp):
        input_shape = (83, 6, 1)
        output_shape = 10
        
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
        
        hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        # Select optimizer    
        # Refer to https://stackoverflow.com/questions/67286051/how-can-i-tune-the-optimization-function-with-keras-tuner
        hp_optimizer = hp.Choice('optimizer', values=['sgd', 'rmsprop', 'adam'])

        if hp_optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=hp_learning_rate)
        elif hp_optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=hp_learning_rate)
        elif hp_optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
    
        model.compile(optimizer=optimizer,
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [16, 32,64]),
            **kwargs,
        )

class CNN_F_1_A_HyperModel(kt.HyperModel):
    def build(self, hp):
        input_shape = (130, 20, 1)
        output_shape = 10
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
        
        hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        # Select optimizer    
        # Refer to https://stackoverflow.com/questions/67286051/how-can-i-tune-the-optimization-function-with-keras-tuner
        hp_optimizer = hp.Choice('optimizer', values=['sgd', 'rmsprop', 'adam'])

        if hp_optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=hp_learning_rate)
        elif hp_optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=hp_learning_rate)
        elif hp_optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
    
        model.compile(optimizer=optimizer,
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [16, 32,64]),
            **kwargs,
        )



