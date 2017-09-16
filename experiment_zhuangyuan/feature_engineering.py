# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import pandas as pd
import data_trans as dt
import common_utils as cu
from sklearn import preprocessing
import xgboost as xgb

def getTrainValid(X, k, num):
    train_index = []
    valid_index = []
    for i in range(len(X)):
        if i % k != num:
            train_index.append(i)
        else:
            valid_index.append(i)
    
    return train_index, valid_index

def runTrain(X_trans, y, fold, fold_index, tarcols):
    ti, vi = getTrainValid(X_trans, fold, fold_index)
    x_train, y_train, x_valid, y_valid = X_trans.iloc[ti,:],y[ti],X_trans.iloc[vi,:],y[vi]
    
    x_train_trim = x_train[tarcols]
    x_valid_trim = x_valid[tarcols]
    d_train = xgb.DMatrix(x_train_trim, label=y_train)
    d_valid = xgb.DMatrix(x_valid_trim, label=y_valid)
    
#    print('Training the model.')
    # xgboost params
    params = {
        'eta': 0.033,
        'max_depth': 4,
        'subsample': 0.80,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'silent': 1
    }
    
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    model = xgb.train(params, d_train, 500, watchlist, early_stopping_rounds=100,
                      verbose_eval=1000)
    
    best = model.best_score
    return best
    

# read train data.
X, y = cu.get_train_data(encode_non_object=True)
tarlist = X.columns
X_trans, propdic = dt.getTransData(X, y, tarlist)

for c in tarlist:
    X_trans[c] = X_trans[c].astype(float)
# get CV from train data.
rlist = []
tarcols = ['calculatedfinishedsquarefeet',]
fold = 10
for i in range(10):
    best = runTrain(X, y, fold, i, tarcols)
    rlist.append([i, best])

for r in rlist:
    print(*r,sep='\t')
    

#查看随机森林各树的构成
ti, vi = getTrainValid(X_trans, fold, 1)
x_train, y_train, x_valid, y_valid = X_trans.iloc[ti,:],y[ti],X_trans.iloc[vi,:],y[vi]

x_train_trim = x_train[tarcols]
x_valid_trim = x_valid[tarcols]
d_train = xgb.DMatrix(x_train_trim, label=y_train)
d_valid = xgb.DMatrix(x_valid_trim, label=y_valid)

#    print('Training the model.')
# xgboost params
params = {
    'eta': 0.033,
    'max_depth': 4,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 1
}

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
model = xgb.train(params, d_train, 500, watchlist, early_stopping_rounds=100,
                  verbose_eval=1000)

best = model.best_score