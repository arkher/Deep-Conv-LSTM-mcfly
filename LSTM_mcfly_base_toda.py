#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from datetime import datetime
from random import random
import pandas as pd
import numpy as np
from mcfly import modelgen, find_architecture, storage
import sys
import os
#os.environ['KERAS_BACKEND'] = 'tensorflow'
import argparse
import keras.backend as K
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



def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)

# def fscore (y,pred):
#     cm = confusion_matrix(y,pred)
#     tn, fp, fn, tp = cm.ravel()
#     pos = tp + fn + 0.0
#     neg = fp + tn + 0.0
#     fscore = float(2*tp)/float(2*tp + fp + fn)
#     return fscore

# def recall (y,pred):
#     cm = confusion_matrix(y,pred)
#     tn, fp, fn, tp = cm.ravel()
#     pos = tp + fn + 0.0
#     neg = fp + tn + 0.0
#     if(tp == 0 and fn == 0):
#         sens = 0
#     else:
#         sens = float(tp)/float(tp + fn)
#     return sens


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

def test_path(_path):
    if not os.path.isfile(_path):
        print("\nThe path '"+_path+"' does not exist! Aborting...\n")
        exit()  

if __name__ == '__main__':
    labels = ['Class 0', 'Class 1']
    class_column = 'CNR' #CNR - Falta_de_Energia
    time_column = ''
    id_column = ''
    path_predict = ''

    #print("\\")
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_path", help="Folder name to save predict results. Default: root")
    parser.add_argument("--class_weight", help="Use or not class weight (1 or 0). Default: 0", type=int)
    parser.add_argument("--load_model", help="Path to load a trained saved model.")
    args,unknown =  parser.parse_known_args()

    if len(sys.argv) < 5:
        sys.exit('Need at least 4 arguments. The order is as follows:\n\
            1.base_type: (1) Talend - (2) FeatureTools;\n\
            2.path_train_class_0; 3.path_train_class_1;\n\
            4.path_test;\n')

    base_type = int(sys.argv[1])
    path_train_0 = sys.argv[2]
    path_train_1 = sys.argv[3]
    path_test = sys.argv[4]


    test_path(path_train_0)
    test_path(path_train_1)
    test_path(path_test)

    if(base_type==1):
        base_name = '3D'
    
    elif(base_type==2):
        base_name = '2D'
    else:
        sys.exit('Wrong base type. Choose:\n\
            (1) 3D - (2) 2D;\n')

    if(args.predict_path!=None):
        path_predict = args.predict_path + "/"

    chunksize = 10**4
    pretrain = True

    df_train_0 = pd.read_csv(path_train_0, chunksize=chunksize)
    df_train_1 = pd.read_csv(path_train_1)
    for i in range(0,5):
        for chunk_train_0 in df_train_0:
            chunk_train = pd.concat([chunk_train_0,df_train_1])
            chunk_train = chunk_train.sample(frac=1).reset_index(drop=True)

            if(base_type==1):
                df_train_X = chunk_train.drop([class_column, time_column], axis = 1)
            else:
                df_train_X = chunk_train.drop([class_column,'CONTA_CONTRATO'], axis = 1)

            df_train_Y = chunk_train[[class_column]]

            X_train = df_train_X.values
            y_train_old = df_train_Y.values

            if(base_type==1):
                X_train = X_train.reshape(int(X_train.shape[0]),31,int(X_train.shape[1]/31))
            else:
                X_train = X_train.reshape(X_train.shape[0],1,int(X_train.shape[1]))

            if(pretrain):
                y_train = to_categorical(y_train_old,2)
                print("Pre Treino")
                porc = int((X_train.shape[0])*0.8)

                X_val = X_train[porc:,:]
                y_val = y_train[porc:,:]

                X_train = X_train[:porc,:]
                y_train = y_train[:porc,:]

                print(X_train.shape)
                print(X_val.shape)
                models = modelgen.generate_models(X_train.shape,
                                                  number_of_classes=2,
                                                  number_of_models =5,
                                                  # metrics = {'f':fmeasure,'p':precision,'r':recall},
                                                  #metrics = {'acc':'accuracy','f':fmeasure,'p':precision,'r':recall},
                                                  model_type='DeepConvLSTM',
                                                  low_lr=2, high_lr=5
                                                  )


                models_to_print = range(len(models))
                for i, item in enumerate(models):
                    if i in models_to_print:
                        model, params, model_types = item
                        print("-------------------------------------------------------------------------------------------------------")
                        print("Model " + str(i))
                        print(" ")
                        print("Hyperparameters:")
                        print(params)
                        print(" ")
                        print("Model description:")
                        model.summary()
                        print(" ")
                        print("Model type:")
                        print(model_types)
                        print(" ")

                

                modelcomparison = "modelcomparison4"
                outputfile = path_predict+modelcomparison+'.json'
                histories, val_metrics, val_losses = find_architecture.train_models_on_samples(X_train, y_train,
                                                                                           X_val, y_val,
                                                                                           models,nr_epochs=3,
                                                                                           subset_size=80000,
                                                                                           verbose=True,
                                                                                           #metric='acc',
                                                                                           outputfile=outputfile)
                print(np.asarray(val_metrics).shape)
                print(val_metrics)
                print(len(histories))
                sys.exit()


                modelcomparisons = pd.DataFrame({'model':[str(params) for model, params, model_types in models],
                                        'train_acc': [history.history['acc'][-1] for history in histories],
                                        'train_loss': [history.history['loss'][-1] for history in histories],
                                        'val_acc': [history.history['val_acc'][-1] for history in histories],
                                        'val_loss': [history.history['val_loss'][-1] for history in histories]
                                        })

                #sys.exit()

                print(np.asarray(val_metrics).shape)
                print(val_metrics)
                print(modelcomparisons)
                sys.exit()
                
                modelcomparisons_csv = path_predict+modelcomparison+'.csv'
                modelcomparisons.to_csv(modelcomparisons_csv)

                print(modelcomparisons)

                best_model_index = np.argmax(val_metrics)
                best_model, best_params, best_model_types = models[best_model_index]
                print('Model type and parameters of the best model:')
                print(best_model_index)
                print(best_model_types)
                print(best_params)
                pretrain = False

                sys.exit()
            else:
                y_train = to_categorical(y_train_old,2)
                print("Treino")

                porc = int((X_train.shape[0])*0.8)

                X_val = X_train[porc:,:]
                y_val = y_train[porc:,:]

                X_train = X_train[:porc,:]
                y_train = y_train[:porc,:]

                print(X_train.shape)
                print(X_val.shape)
                if(args.class_weight==1):
                    y_integers = np.argmax(y_train, axis=1)
                    class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
                    d_class_weights = dict(enumerate(class_weights))

                    history = best_model.fit(X_train, y_train,
                                  epochs=1, validation_data=(X_val, y_val),class_weight=d_class_weights)
                else:
                    history = best_model.fit(X_train, y_train,
                                  epochs=1, validation_data=(X_val, y_val))

                #while os.path.isfile(path_predict+'bestmodel2.h5')
                best_model_file = path_predict+'bestmodel3.h5'
                best_model.save(best_model_file)

   
print('DONE')
        


