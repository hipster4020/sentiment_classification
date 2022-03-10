import numpy as np
import torch
from datasets import load_metric

#from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(p, r):
    f1_metric = load_metric("f1")
    f1_results = f1_metric.compute(predictions=p, references=r, average="micro")
    
    accuracy = load_metric("accuracy")
    
    return {"f1score" : f1_results, "accuracy" : accuracy}

# def compute_metrics(p):    
#     pred, labels = p
#     pred = np.argmax(pred, axis=1)
#     accuracy = accuracy_score(y_true=labels, y_pred=pred)
#     f1 = f1_score(y_true=labels, y_pred=pred, average='macro')
#     return {"accuracy": accuracy, "macro f1": f1} 


#loss = torch.nn.BCELoss()
