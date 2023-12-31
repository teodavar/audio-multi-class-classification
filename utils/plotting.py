"""
Created on Tue Apr 11 15:24:26 2023

@author: davarakis
"""
import matplotlib.pyplot as plt

# Plots the training/Validation loss plot
# Plots the training/validation accuracy plot
def plot_training(train_losses, train_accs, val_losses, val_accs, model_name="", return_fig=True):
        '''
        Plot losses and accuracy over the training process.

        train_losses (list): List of training losses over training.
        train_accs (list): List of training accuracies over training.
        val_losses (list): List of validation losses over training.
        val_accs (list):List of validation accuracies over training.
        model_name (str): Name of model as a string. 
        return_fig (Boolean): Whether to return figure or not. 
        '''
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
        ax1.plot(val_losses, label='Validation loss')
        ax1.plot(train_losses, label="Training loss")
        ax1.set_title('Loss over training for {}'.format(model_name), fontsize=20)
        ax1.set_xlabel("epoch",fontsize=18)
        ax1.set_ylabel("loss",fontsize=18)
        ax1.legend()

        ax2.plot(val_accs, label='Validation accuracy')
        ax2.plot(train_accs, label='Training accuracy')
        ax2.set_title('Accuracy over training for {}'.format(model_name), fontsize=20)
        ax2.set_xlabel("epoch",fontsize=18)
        ax2.set_ylabel("accuracy",fontsize=18)
        ax2.legend()

        fig.tight_layout() 
        if return_fig:
                return fig

# Credits to https://stackoverflow.com/users/16504277/nico
# Credits to https://github.com/intentodemusico/py2Tex     

# Converts the model.summary to a latex table       
def m2tex(model,modelName):
    stringlist = []
    model.summary(line_length=70, print_fn=lambda x: stringlist.append(x))
    del stringlist[1:-4:2]
    del stringlist[-1]
    for ix in range(1,len(stringlist)-3):
        tmp = stringlist[ix]
        stringlist[ix] = tmp[0:31]+"& "+tmp[31:59]+"& "+tmp[59:]+"\\\\ \hline"
    stringlist[0] = "Model: {} \\\\ \hline".format(modelName)
    stringlist[1] += " \hline"
    stringlist[-4] += " \hline"
    stringlist[-3] += " \\\\"
    stringlist[-2] += " \\\\"
    stringlist[-1] += " \\\\ \hline"
    prefix = ["\\begin{table}[]", "\\begin{tabular}{lll}"]
    suffix = ["\end{tabular}", "\caption{{Model summary for {}.}}".format(modelName), "\label{tab:model-summary}" , "\end{table}"]
    stringlist = prefix + stringlist + suffix 
    out_str = " \n".join(stringlist)
    out_str = out_str.replace("_", "\_")
    out_str = out_str.replace("#", "\#")
    print(out_str)