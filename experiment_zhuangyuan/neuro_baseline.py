# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import pandas as pd
import common_utils as cu
from sklearn import preprocessing

def getBins(X, attri, num):
    tarseri = X[attri][X[attri]!=-1]
    #
    mnum = int(len(tarseri)/10)
    num = min([num, mnum])
    
    seri = pd.qcut(tarseri, num, duplicates='drop')
    vc = seri.value_counts()
    bins = sorted([ind.left for ind in vc.index]) + [10**20]
    bins[0] = -10**20
    
    return bins

def getBinQuan(aseri, bseri):
    avg = bseri.mean()
    df = pd.DataFrame({'a':aseri, 'b':bseri})
    pt = pd.pivot_table(df,index='a',values='b',aggfunc=np.mean)
    vnum = df.a.value_counts()
    vnumdic = dict(zip(list(vnum.index),list(vnum.values)))
    
    klist = list(pt.index)
    vlist = list(pt.b)
    
    rdic = dict(zip(klist, vlist))

    for k in rdic.keys():
        rdic[k] = (100.0*avg + vnumdic[k]*rdic[k])/(100.0+vnumdic[k])
    
    return rdic

def getTransData(X, tarlist):
    X_trans = pd.DataFrame()
    propdic = {}
    for k in X.columns:
        if not k in tarlist:
            X_trans[k] = X[k]
            continue
        
        valnum = X[k].value_counts().shape[0]

        if valnum > 20:
            try:
                bins = getBins(X, k, 30)
                if bins[1] > 0:
                    bins.insert(1, 0.0)
            except:
                continue
            aseri = pd.cut(X[k],bins,include_lowest=True,labels=range(len(bins)-1))    
        else:
            aseri = X[k]
            bins = []
        
        mapdic = getBinQuan(aseri, y)
        X_trans[k] = aseri.apply(lambda x:mapdic[x])
        propdic[k] = propdic.get(k, {})
        propdic[k]['bins'] = bins
        propdic[k]['mapdic'] = mapdic
    
    return X_trans, propdic

def getTransTest(X, propdic):
    X_trans = pd.DataFrame()
    for k in X.columns:
        if not k in propdic.keys():
            X_trans[k] = X[k]
        else:
            if len(propdic[k]['bins']) == 0:
                aseri = X[k]
            else:
                aseri = pd.cut(X[k], propdic[k]['bins'], include_lowest=True, labels=range(len(propdic[k]['bins'])-1))
            X_trans[k] = [float(propdic[k]['mapdic'].get(x,0.01)) for x in aseri]
#                X_trans[k] = [float(x) for x in aseri]
    
    return X_trans

def showDataValues(X_trans):
    with open('../../data/result.data', 'w') as outfile:
        for c in X_trans.columns:
            outfile.write(c+'\n')
            outfile.write(str(X_trans[c].value_counts()))
            outfile.write('\n')
    
    with open('../../data/result.data', 'w') as outfile:
        for c in X.columns:
            outfile.write(c+'\n')
            outfile.write(str(X[c].value_counts()))
            outfile.write('\n')
        
def neuroforcast():
    model = Sequential([
    Dense(32, input_shape=(100,)),
    Activation('relu'),
    Dense(1),
    Activation('linear'),
    ])
    
    # For a mean squared error regression problem
    model.compile(optimizer='rmsprop',
                  loss='mse')
    # Generate dummy data
    
    data = np.random.random((1000, 100))
    wlist = np.random.random(100)
    labels = 0.1*np.random.random(1000)
    for i in range(100):
        labels = labels + wlist[i]*data[:, i]
    
    #labels = np.random.randint(2, size=(1000, 1))
    
    # Train the model, iterating on the data in batches of 32 samples
    model.fit(data, labels, epochs=10, batch_size=32)
    
    y_pred = model.predict(data)
    
    df = pd.DataFrame({1:list(y_pred[:,0]),2:list(labels)})
    df.to_clipboard()

# read train data.
X, y = cu.get_train_data(encode_non_object=True)
tarlist = X.columns#['longitude', 'yearbuilt', 'taxamount']
X_trans,propdic = getTransData(X, tarlist)

x_train, y_train, x_holdout, y_holdout = cu.get_cv(X_trans, y)

model = Sequential([
Dense(32, input_shape=(53,)),
Activation('relu'),
Dense(1),
Activation('linear'),
])

# For a mean squared error regression problem
model.compile(optimizer='rmsprop',
              loss='mse')

model.fit(x_train.values, y_train, epochs=10, batch_size=32)

y_pred = model.predict(x_holdout.values)

model.fit(X_trans.values, y, epochs=10, batch_size=32)
# read test data.
T = cu.get_test_data(encode_non_object=True)
T_trans = getTransTest(T, propdic)
# predict result.
print('Predicting.')
y_pred = model.predict(T_trans[X_trans.columns].values)

# write result.
cu.write_result(y_pred)
