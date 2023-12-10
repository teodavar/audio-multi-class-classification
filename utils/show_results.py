"""
Created on Tue Apr 11 15:24:26 2023

@author: davarakis
"""
import json
from utils.plotting import plot_training
import pandas as pd

# Summarizes the results of the models' training

# Reads a list of logfiles and prints for each logfile the best epoch and
# the training plots. It also aggregates the training results into a dataframe.

# At the end, prints the model that performs best on validation set and the 
# the best 3 performing models.

def show_results(log_filenames):
    models = []
    train_accs = []
    val_accs = []
    best_epochs = []
    for log_file in log_filenames:
        print(log_file)
        md_name = log_file.split('_')
        print(md_name)
        if (len(md_name) == 7):
            tuned = True
        else:
            tuned = False
        md_name = md_name[0] + "_" + md_name[1] + "_" + md_name[2] + "_" + md_name[3]
        models.append(md_name)
        print(md_name)

        if tuned:
                md_name = md_name + "_tuned"
                print(md_name)

                full_log_filename = "logs/"+md_name+"/"+log_file
        else:
                full_log_filename = "logs/"+md_name.lower()+"/"+log_file
        #print(full_log_filename)
        with open(full_log_filename, 'r') as f:
            print(full_log_filename)
            val_json = json.load(f)
            best_val_epoch = val_json["best_val_epoch"]
            print("best_val_epoch: ", best_val_epoch)
            train_accs.append("{0:.2f}".format(val_json["train_accs"][best_val_epoch-1]))
            val_accs.append("{0:.2f}".format(val_json["val_accs"][best_val_epoch-1]))
            best_epochs.append(best_val_epoch)
            #print(best_val_epoch)
            #print("BEST TRAIN ACC: ","{0:.2f}".format(val_json["train_accs"][best_val_epoch]) )
            #print("BEST VAL ACC: ","{0:.2f}".format(val_json["val_accs"][best_val_epoch]) )

            plot_training(val_json["train_losses"],
                  val_json["train_accs"],
                  val_json["val_losses"], 
                  val_json["val_accs"],
                  model_name=md_name,
                  return_fig=False)

    results_df = pd.DataFrame({"model":models, 
                            "training accuracy":train_accs,
                            "validation accuracy":val_accs,
                            "best_val_epoch":best_epochs,
                                  })
    print(results_df)

    max_value = max(val_accs)
    max_index = val_accs.index(max_value)
    best = models[max_index]
    print("The model that performs best on validation set is: ", models[max_index])
    
    best_3 = results_df.sort_values(['validation accuracy'],ascending = False).head(3) 
    return results_df, best, best_3
    
# for testing
'''
#log_filenames = ["CNN_F_1_A_TUNED_260423_092716.json", "CNN_S_6_A_TUNED_260423_094718.json"]
log_filenames = ["CNN_F_2_N_260423_073101.json", "CNN_S_1_N_260423_072724.json"]

results_df, best, best_3 = show_results(log_filenames) 
'''
