import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(p):    
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred, average='macro')
    return {"accuracy": accuracy, "macro f1": f1} 


loss = torch.nn.BCELoss()
