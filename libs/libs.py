import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import seaborn as sns


def load_data(data_dir):
    data = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            with open(os.path.join(root, file)) as f:
                file_data = np.loadtxt(open(os.path.join(root, file)).readlines()[:-1], usecols=(0,1))
                # file_data = np.transpose(file_data)
                file_data = file_data-file_data[0,0]             
                data.append(file_data)
    data = np.array(data)
    return data

def calculate_metrics(y_true, y_pred,):
    # calculate the confusion matrix, the accuracy, and the precision and recall 
    # binary trick
    y_pred_am = np.where(y_pred > 0.5, 1, 0)
    cm = confusion_matrix(y_true, y_pred_am, normalize='true')
    # compute precision matrix
    

    accuracy = accuracy_score(y_true, y_pred_am)
    precision = precision_score(y_true, y_pred_am, average='macro')
    recall = recall_score(y_true, y_pred_am, average='macro')
    f1 = f1_score(y_true, y_pred_am, average='macro')

    return cm, accuracy, precision, recall, f1
    
def log_metrics(y_true, y_pred, output_folder="", label_names=["Background", "Signal"]):
    cm, accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred)
    print("Confusion Matrix")
    print(cm)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(output_folder+f"metrics.txt", "a") as f:
        f.write("Confusion Matrix\n")
        f.write(str(cm)+"\n")
        f.write("Accuracy: "+str(accuracy)+"\n")
        f.write("Precision: "+str(precision)+"\n")
        f.write("Recall: "+str(recall)+"\n")
        f.write("F1: "+str(f1)+"\n")
    # save confusion matrix 
    plt.figure(figsize=(10,10))
    plt.title("Confusion matrix", fontsize=28)
    sns.heatmap(cm, annot=True, cmap="YlGnBu", xticklabels=label_names, yticklabels=label_names, annot_kws={"fontsize": 20})
    plt.ylabel('True label', fontsize=28)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Predicted label', fontsize=28)
    plt.savefig(output_folder+f"confusion_matrix.png")
    plt.clf()
    # Binarize the output
    y_test = label_binarize(y_true, classes=np.arange(len(label_names)))
    n_classes = y_test.shape[1]
    plt.figure()
    fpr, tpr, _ = roc_curve(y_true[:], y_pred[:])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label='ROC curve (area = {0:0.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=28)
    plt.ylabel('True Positive Rate', fontsize=28)
    plt.title('ROC curve', fontsize=28)
    plt.legend(loc="lower right", fontsize=20)
    plt.savefig(output_folder+"roc_curve.png")
    plt.clf()
    # create an histogram of the predictions
    y_true = np.reshape(y_true, (y_true.shape[0],))
    bkg_preds = y_pred[y_true < 0.5]
    sig_preds = y_pred[y_true > 0.5]
    print("Background predictions: ", bkg_preds.shape)
    print("Signal predictions: ", sig_preds.shape)
    plt.hist(bkg_preds, bins=50, range=(0,1), alpha=0.5, label=f'{label_names[0]} (n={bkg_preds.shape[0]})')
    plt.hist(sig_preds, bins=50, range=(0,1), alpha=0.5, label=f'{label_names[1]} (n={sig_preds.shape[0]})')
    plt.legend(loc='upper right')
    plt.xlabel('Prediction')
    plt.ylabel('Counts')
    plt.title('Predictions')
    plt.savefig(output_folder+f"predictions.png")
    plt.clf()
    return



