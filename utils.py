from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
import numpy as np
import torch
sigmoid = torch.nn.Sigmoid()

def get_perfrom(ground_truth, predictions, probs):
    roc_auc_scores = []
    average_precision_scores = []
    if len(list(set(ground_truth)))==1:
        roc_auc = 'Nan'
        pr_auc = 'Nan'
    else:
        for i in range(9):
            y_true = [1 if x == i else 0 for x in ground_truth]
            y_pred = sigmoid(torch.tensor(probs[:, i])).numpy()
            if sum(y_true) == 0:
                continue
            else:
                roc_auc_scores.append(roc_auc_score(y_true, y_pred))
                average_precision_scores.append(average_precision_score(y_true, y_pred))
        
        roc_auc = sum(roc_auc_scores) / len(roc_auc_scores)
        pr_auc = sum(average_precision_scores) / len(average_precision_scores)

    accuracy = accuracy_score(ground_truth, predictions)
    balanced_ACC = balanced_accuracy_score(ground_truth, predictions)
    mcc = matthews_corrcoef(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, average='macro')
    recall = recall_score(ground_truth, predictions, average='macro')
    f1 = f1_score(ground_truth, predictions, average='macro')
    return [roc_auc, pr_auc, accuracy, balanced_ACC, mcc, precision, recall, f1]
