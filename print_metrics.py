import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def print_metrics(y,pred):
    cm = confusion_matrix(y,pred)
    tn, fp, fn, tp = cm.ravel()
    pos = tp + fn + 0.0
    neg = fp + tn + 0.0
    acc = float(tp + tn)/float(pos + neg)
    prec = float(tp)/float(tp + fp)
    sens = float(tp)/float(tp + fn)
    spec = float(tn)/float((tn + fp))
    fscore = float(2*tp)/float(2*tp + fp + fn)
    print("Acc\t\tPrec\t\tSens\t\tSpec\t\tFscore\t\tTP\tFN\tFP\tTN")
    print("{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:d}\t{:d}\t{:d}\t{:d}".format(acc,prec,sens,spec,fscore,tp,fn,fp,tn))

df_test = pd.read_csv("Falta_de_Energia\\2D_original_predicts_processos.csv")
df_pred = pd.read_csv("Falta_de_Energia\\2D_predicts_processos.csv")

original = df_test.values
prediction = df_pred.values
print(original.shape)
print(prediction.round().shape)
print_metrics(original,prediction.round())