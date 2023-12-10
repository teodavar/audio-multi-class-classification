"""
Created on Tue Apr 11 15:24:26 2023

@author: davarakis
"""
import tensorflow as tf
import keras_tuner as kt

import pandas as pd

import audio_processing as pr 
import run_models as rm


from hypermodels import CNN_F_1_A_HyperModel
from hypermodels import CNN_S_6_A_HyperModel

import numpy as np
import json
from sklearn.model_selection import train_test_split
from utils.plotting import plot_training
import os
import time

# The best performing models from each workflow in fined tuned.
# Keras-Tuner is used

# Prints the results from the search of the hyperparameters space
def my_show_results(tuner, max_trials):
    best_trials = tuner.oracle.get_best_trials(max_trials)
    trials = []
    best_optimizers = []
    best_lrs = []
    best_bsizes = []
    best_scores = []
    
    for trial in best_trials:
        trials.append(trial.trial_id)
        best_scores.append(trial.score)
    
        for hp, value in trial.hyperparameters.values.items():
            
            print(f"{hp}:", value)
            if hp == "optimizer":
              best_optimizers.append(value)
            if hp == "learning_rate":
              best_lrs.append(value)
            if hp == "batch_size":
              best_bsizes.append(value)
    data = {
      "trial": trials,
      "optimizer": best_optimizers,
      "learning_rate": best_lrs,
      "batch_size": best_bsizes,
      "scores": best_scores
    }
    results = pd.DataFrame(data)
    #print(df) 
    return results

# Models to be fined tunned are : CNN_S_6_A or CNN_F_1_A
def fine_tuning(model_name, log_filename):
    
    if (model_name == 'CNN_S_6_A'):
        # Read increased dataset
        features_filename = "gtzan/features-3sec"
        labels_filename = "gtzan/labels-3sec"
        
        features = rm.read_list_fromfile(features_filename)
        print('Length of read features is: ', len(features))
        
        labels = rm.read_list_fromfile(labels_filename)
        print('Length of read labels is: ', len(labels))
        
        features_train, labels_train, features_val, labels_val, features_test, labels_test = pr.prepareStatisticsDataset(features, labels)
        features_train = features_train.reshape(features_train.shape[0], 83, 6, 1)
        features_val = features_val.reshape(features_val.shape[0], 83, 6, 1)
    else:
        # Read increased dataset
        filename ="gtzan/second-all-3secs.json"

        with open(filename) as f:
            data = json.load(f)
            f.close()
        # turn data into numpy arrays
        inputs = np.array(data["mfcc"])
        labels = np.array(data["labels"])
        inputs_training, inputs_test, labels_training, labels_test = train_test_split(inputs, labels, test_size=0.2, shuffle=True, random_state=2023)
        inputs_train, inputs_val, labels_train, labels_val = train_test_split(inputs_training, labels_training, test_size=0.2, shuffle=True, random_state=2023)
        print("Length of inputs_train:", inputs_train.shape)
        print("Length of labels_train:", labels_train.shape)
        print("Length of inputs_val:", inputs_val.shape)
        print("Length of labels_val:", labels_val.shape)

        
    # Correct values 
    # max_trails = 10
    max_trials = 10
    # NUM_EPOCHS = 100 
    NUM_EPOCHS = 100
    
    # Instantiate the RandomSearch tuner and specify:
    # the val_accuracy to be the objective to be optimized
    if (model_name == 'CNN_S_6_A'):
        my_tuner = kt.RandomSearch(
            CNN_S_6_A_HyperModel(),
            objective="val_accuracy",
            max_trials=max_trials,
            overwrite=True,
            directory="logs",
            project_name=model_name+"_TUNNING",
        )
    else:
        my_tuner = kt.RandomSearch(
            CNN_F_1_A_HyperModel(),
            objective="val_accuracy",
            max_trials=max_trials,
            overwrite=True,
            directory="logs",
            project_name=model_name+"_TUNNING",
        )
    
    # Display search space summary
    my_tuner.search_space_summary()
    
    # Create a callback to stop training early after reaching a certain value for the validation loss
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    
    # Run the hyperparameter search
    if (model_name == 'CNN_S_6_A'):
        my_tuner.search(features_train, labels_train, epochs=NUM_EPOCHS, validation_data=(features_val, labels_val), callbacks=[stop_early], verbose=0)
    else:
        my_tuner.search(inputs_training, labels_training, epochs=NUM_EPOCHS, validation_split=0.2, callbacks=[stop_early], verbose=0)
    
    # Show the hyperparemeters seach space results
    print("---------------------------------------------------------------")
    results_summary = my_tuner.results_summary()
    print(results_summary)
    print("---------------------------------------------------------------")   
    
    df_results = my_show_results(my_tuner, max_trials)
    print(df_results)
    
    # Get the optimal hyperparameters
    best_hps=my_tuner.get_best_hyperparameters(num_trials=1)[0]
    best_learning_rate = best_hps.get('learning_rate')
    best_batch_size = best_hps.get('batch_size')
    best_optimizer = best_hps.get('optimizer')
    print("Best optimizer is: ", best_optimizer)
    print("Optimal learning rate is: ", best_learning_rate)
    print("Optimal batch size is: ", best_batch_size)
    print("Build the model with the optimal hyperparameters and train it.")

    # Find the optimal number of epochs to train the model with the hyperparameters obtained from the search.
    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    
    model = my_tuner.hypermodel.build(best_hps)
    
    #history = model.fit(inputs_training, labels_training, epochs=NUM_EPOCHS, validation_split=0.2)
    if (model_name == 'CNN_S_6_A'):
        history = model.fit(x=features_train.tolist(),y=labels_train.tolist(),
                                validation_data=(features_val.tolist() , labels_val.tolist()), 
                                epochs=NUM_EPOCHS)
    else:
        history = model.fit(inputs_training, labels_training, epochs=NUM_EPOCHS, validation_split=0.2)
    
    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))
        
    # Re-instantiate the hypermodel and train it with the optimal number of epochs from above.
    hypermodel = my_tuner.hypermodel.build(best_hps)
    # Retrain the model
    if (model_name == 'CNN_S_6_A'):
        history = hypermodel.fit(x=features_train.tolist(),y=labels_train.tolist(),
                                validation_data=(features_val.tolist() , labels_val.tolist()), 
                                epochs=best_epoch)
    else:
        history = hypermodel.fit(inputs_training, labels_training, epochs=best_epoch, validation_split=0.2)
    
    # Save the final model
    save_filename = "savedmodels/"+ model_name+"_tunned.h5" 
    hypermodel.save(save_filename)     
    
    # Get training performance results
    tuning_results = {
      "summary": df_results,
      "best_optimizer": best_optimizer,
      "best_learning_rate": best_learning_rate,
      "best_batch_size": best_batch_size,
      "best_epoch": best_epoch
    }
    #print('HISTORY: ',  history.history)
    
    training_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    
    logname = model_name + '_tuned'
    fig = plot_training(training_loss, training_accuracy,validation_loss, validation_accuracy)
    fig.savefig(os.path.join("figs", "{}_training_vis".format(logname)))

    # Some log information to help you keep track of your model information. 
    #print("PARAMS:", params["lr"])
    logs ={
            "model": logname,
            "train_losses": training_loss,
            "train_accs": training_accuracy,
            "val_losses": validation_loss,
            "val_accs": validation_accuracy,
            "best_val_epoch": int(np.argmax(validation_accuracy)+1),
            "lr": best_learning_rate,
            "num_epochs": best_epoch,
            "batch_size":best_batch_size
        }

    print(" --- log_filename:" , log_filename)
    print(" ---- logname:" , logname)
    new_log_dir = 'logs/'+logname
    if not os.path.exists(new_log_dir): os.makedirs(new_log_dir)
    #log_dir= 'logs/'+logname
    full_log_filename = 'logs/'+logname+'/'+log_filename
    with open(full_log_filename, 'w') as f:
        json.dump(logs, f)
    
    
    return tuning_results

# Currently model_name is : CNN_S_6_A or CNN_F_1_A
# for testing!!!
'''
tuned_log_filenames = []
model_name = "CNN_F_1_A_TUNED"
start_time = time.strftime("%d%m%y_%H%M%S")
tuned_log_filename = "{}_{}.json".format(model_name,  start_time) 
tuned_log_filenames.append(tuned_log_filename)

tuning_results_cnn_f_1_a = fine_tuning("CNN_F_1_A", tuned_log_filename)
'''


