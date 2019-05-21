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


def print_metrics(y,pred, caminho):
    cm = confusion_matrix(y,pred)
    tn, fp, fn, tp = cm.ravel()
    pos = tp + fn + 0.0
    neg = fp + tn + 0.0
    acc = float(tp + tn)/float(pos + neg)
    if(float(tp + fp) == 0.0):
        prec = 0.0
    else:
        prec = float(tp)/float(tp + fp)
    sens = float(tp)/float(tp + fn)
    spec = float(tn)/float((tn + fp))
    fscore = float(2*tp)/float(2*tp + fp + fn)
    print("Acc\t\tPrec\t\tSens\t\tSpec\t\tFscore\t\tTP\tFN\tFP\tTN")
    print("{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:d}\t{:d}\t{:d}\t{:d}".format(acc,prec,sens,spec,fscore,tp,fn,fp,tn))
    with open(caminho + '\\metricsLSTM' + '.txt', 'w') as metricsCliente:
        print('acc: {}\n'.format(acc), file = metricsCliente)
        print('prec : {}\n'.format(prec), file = metricsCliente)
        print('sens : {}\n'.format(sens), file = metricsCliente)
        print('spec: {}\n'.format(spec), file = metricsCliente)
        print('fscore : {}\n'.format(fscore), file = metricsCliente)
        print('tp : {}\n'.format(tp), file = metricsCliente)
        print('fn : {}\n'.format(fn), file = metricsCliente)
        print('fp : {}\n'.format(fp), file = metricsCliente)
        print('tn : {}\n'.format(tn), file = metricsCliente)
        

def test_path(_path):
    if not os.path.isfile(_path):
        print("\nThe path '"+_path+"' does not exist! Aborting...\n")
        exit()  

if __name__ == '__main__':
    labels = ['Class 0', 'Class 1']
    class_column = 'Class' #Processo - CNR - Falta_de_Energia
    time_column = ''
    id_column = ''
    path_predict = ''
    epocas = 5

    #print("\\")
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_path", help="Folder name to save predict results. Default: root")
    parser.add_argument("--class_weight", help="Use or not class weight (1 or 0). Default: 0", type=int)
    parser.add_argument("--load_model", help="Path to load a trained saved model.")
    parser.add_argument("--epochs", help="Number of epochs (fit on the best model). Default: 5", type=int)
    parser.add_argument("--class_column", help="Name of column class. Default: Class")
    args,unknown =  parser.parse_known_args()

    if len(sys.argv) < 4:
        sys.exit('Need at least 3 arguments. The order is as follows:\n\
            1.base_type: (1) 3D - (2) 2D;\n\
            2.path_train;\n\
            3.path_test;\n')

    base_type = int(sys.argv[1])
    path_train = sys.argv[2]
    path_test = sys.argv[3]

    test_path(path_train)
    test_path(path_test)

    if(base_type==1):
        base_name = '3D'
    
    elif(base_type==2):
        base_name = '2D'
    else:
        sys.exit('Wrong base type. Choose:\n\
            (1) 3D - (2) 2D;\n')

    if(args.predict_path!=None):
        path_predict = args.predict_path + "\\"

    if(args.class_column!=None):
        class_column = args.class_column

    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)

    df_train = df_train.sample(frac=1).reset_index(drop=True)

    if(base_type==1):
        df_train_X = df_train.drop([class_column, time_column], axis = 1)
        df_test_X = df_test.drop([class_column, time_column], axis = 1)
    else:
        df_train_X = df_train.drop([class_column], axis = 1) #'CONTA_CONTRATO',Qntd_Processo, CONTA_CONTRATO, 'PROC_TRAFO', 'PERDA_TRAFO', 'QTD_PROC_TRAFO', 'QTD_PERDA_TRAFO'
        df_test_X = df_test.drop([class_column], axis = 1) #Qntd_Processo, CONTA_CONTRATO, 'PROC_TRAFO', 'PERDA_TRAFO', 'QTD_PROC_TRAFO', 'QTD_PERDA_TRAFO'

    df_train_Y = df_train[[class_column]]
    df_test_Y = df_test[[class_column]]

    X_train = df_train_X.values
    y_train_old = df_train_Y.values

    X_test = df_test_X.values
    y_test_old = df_test_Y.values

    if(base_type==1):
        X_train = X_train.reshape(int(X_train.shape[0]),31,int(X_train.shape[1]/31))
        X_test = X_test.reshape(int(X_test.shape[0]),31,int(X_test.shape[1]/31))
    else:
        X_train = X_train.reshape(X_train.shape[0],1,int(X_train.shape[1]))
        X_test = X_test.reshape(X_test.shape[0],1,int(X_test.shape[1]))

    y_train = to_categorical(y_train_old)
    y_test = to_categorical(y_test_old)

    porc = int((X_train.shape[0])*0.8)

    X_val = X_train[porc:,:]
    y_val = y_train[porc:,:]

    X_train = X_train[:porc,:]
    y_train = y_train[:porc,:]

    print(y_train.shape)
    print(y_val.shape)
    print(y_test.shape)

    if(args.epochs!=None):
            epocas = args.epochs
  
    if(args.load_model==None):
        models = modelgen.generate_models(X_train.shape,
                                          number_of_classes=2,
                                          number_of_models =5,
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


        outputfile = path_predict+'modelcomparison.json'
        histories, val_accuracies, val_losses = find_architecture.train_models_on_samples(X_train, y_train,
                                                                                   X_val, y_val,
                                                                                   models,nr_epochs=epocas,
                                                                                   subset_size=10000,
                                                                                   verbose=True,
                                                                                   outputfile=outputfile)


        modelcomparisons = pd.DataFrame({'model':[str(params) for model, params, model_types in models],
                               'train_acc': [history.history['acc'][-1] for history in histories],
                               'train_loss': [history.history['loss'][-1] for history in histories],
                               'val_acc': [history.history['val_acc'][-1] for history in histories],
                               'val_loss': [history.history['val_loss'][-1] for history in histories]
                               })

        modelcomparisons_csv = path_predict+'modelcomparisons.csv'
        modelcomparisons.to_csv(modelcomparisons_csv)

        print(modelcomparisons)

        best_model_index = np.argmax(val_accuracies)
        best_model, best_params, best_model_types = models[best_model_index]
        print('Model type and parameters of the best model:')
        print(best_model_index)
        print(best_model_types)
        print(best_params)

        if(args.class_weight==1):
            y_integers = np.argmax(y_train, axis=1)
            class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
            d_class_weights = dict(enumerate(class_weights))

            history = best_model.fit(X_train, y_train,
                          epochs=epocas, validation_data=(X_val, y_val),class_weight=d_class_weights)
        else:
            history = best_model.fit(X_train, y_train,
                          epochs=epocas, validation_data=(X_val, y_val))

        best_model_file = path_predict+'bestmodel.h5'
        best_model.save(best_model_file)

    else:
        test_path(args.load_model)
        best_model = load_model(args.load_model)
        history = best_model.fit(X_train, y_train,
                          epochs=epocas, validation_data=(X_val, y_val))
        best_model_file = path_predict+'bestmodel2.h5'
        best_model.save(best_model_file)


    probs = best_model.predict_proba(X_test,batch_size=1)
    probs_mean = np.mean(probs, axis=0)
    logloss_test = log_loss(y_test,probs)
    np.savetxt(path_predict+"probs_all_processos.csv",probs,delimiter=',')
    #np.savetxt(path_predict+"probs_mean_processos.csv",probs_mean,delimiter=',')
    #print(probs_mean)
    print(logloss_test)

    

    prediction = best_model.predict(X_test, verbose=True)
    prediction = prediction[:,1]
    #print(prediction)
    df_predicts =  pd.DataFrame(data={'predicts':prediction})
    df_predicts.to_csv(path_predict+base_name+"_predicts_processos.csv", index=False)
    original = y_test[:,1]
    #print(original)
    df_original = pd.DataFrame(data={'predicts':original})
    df_original.to_csv(path_predict+base_name+"_original_predicts_processos.csv", index=False)
    print_metrics(original,prediction.round(),path_predict)

