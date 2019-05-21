from datetime import datetime
from random import random
import pandas as pd
import numpy as np
from mcfly import modelgen, find_architecture, storage
import sys
import os
#os.environ['KERAS_BACKEND'] = 'tensorflow'
import argparse

from keras.models import Sequential  
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.recurrent import LSTM
from keras import metrics
from keras.models import load_model
from keras.utils import to_categorical, multi_gpu_model
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

path_model = 'C:/Users/usuario/Documents/sprint8/LSTM/CNR_Completa_result/bestmodel3.h5'
path_test = 'C:/Users/usuario/Documents/sprint8/CNR_COMPLETA/TEST.csv'
path_predict = 'C:/Users/usuario/Documents/sprint8/LSTM/CNR_Completa_result/'
def print_metrics(y,pred):
	cm = confusion_matrix(y,pred)
	tn, fp, fn, tp = cm.ravel()
	pos = tp + fn + 0.0
	neg = fp + tn + 0.0
	acc = float(tp + tn)/float(pos + neg)
	if(tp == 0 and fp == 0):
	    prec = 0
	else:
	    prec = float(tp)/float(tp + fp)
	if(tp == 0 and fn == 0):
	    sens = 0
	else:
	    sens = float(tp)/float(tp + fn)
	if(tn == 0 and fp == 0):
	    spec = 0
	else:
	    spec = float(tn)/float((tn + fp))
	fscore = float(2*tp)/float(2*tp + fp + fn)
	print("Acc\t\tPrec\t\tSens\t\tSpec\t\tFscore\t\tTP\tFN\tFP\tTN")
	print("{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:d}\t{:d}\t{:d}\t{:d}".format(acc,prec,sens,spec,fscore,tp,fn,fp,tn))

class_column = 'CNR'


#df_test = pd.read_csv(path_test)
model = load_model(path_model)
#df_test_X = df_test.drop([class_column, 'CONTA_CONTRATO'], axis = 1) #Qntd_Processo, CONTA_CONTRATO, 'PROC_TRAFO', 'PERDA_TRAFO', 'QTD_PROC_TRAFO', 'QTD_PERDA_TRAFO'
#df_test_Y = df_test[[class_column]]


#X_test = df_test_X.values
#y_test = df_test_Y.values
#X_test = X_test.reshape(X_test.shape[0],1,int(X_test.shape[1]))
#y_test = to_categorical(y_test)


probas = np.array([])
predictions = np.array([])
y_true_categrorical = np.array([])
for chunk_test in (pd.read_csv(path_test, chunksize=10000)):
	#chunk_test = pd.read_csv(path_test)
	
	df_test_X = chunk_test.drop([class_column,'CONTA_CONTRATO'], axis = 1) #############################################################################################################
	df_test_Y = chunk_test[[class_column]]

	X_test = df_test_X.values
	y_test = df_test_Y.values

	X_test = X_test.reshape(X_test.shape[0],1,int(X_test.shape[1]))

	y_test = to_categorical(y_test,2)
	if(y_true_categrorical.shape[0]==0):
		y_true_categrorical = y_test
	else:
		y_true_categrorical = np.concatenate((y_true_categrorical, y_test))

	probs = model.predict_proba(X_test,batch_size=5000)
	if(probas.shape[0]==0):
		probas = probs
	else:
		probas = np.concatenate((probas,probs))


	prediction = model.predict(X_test, verbose=True)
	prediction = prediction[:,1]	
	
	if(predictions.shape[0]==0):
		predictions = prediction
	else:
		predictions = np.concatenate((predictions,prediction))



        #print(prediction)
df_predicts =  pd.DataFrame(data={'predicts':predictions})
df_predicts.to_csv(path_predict+"_predicts_"+class_column+"_base_toda.csv", index=False)
original = y_true_categrorical[:,1]
    #print(original)
df_original = pd.DataFrame(data={'predicts':original})
df_predicts.to_csv(path_predict+"_y_true_"+class_column+"._base_todacsv", index=False)
print_metrics(original,predictions.round())

probs_mean = np.mean(probas, axis=0)

logloss_test = log_loss(y_true_categrorical,probas)
np.savetxt(path_predict+"probs_all_processos.csv",probas,delimiter=',')
    #np.savetxt(path_predict+"probs_mean_processos.csv",probs_mean,delimiter=',')
    #print(probs_mean)
print(logloss_test)
