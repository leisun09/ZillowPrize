# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

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

def getTransData(X, y, tarlist):
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

if __name__ == '__main__':
    df = pd.DataFrame()
    df['x'] = range(100)
    df['y'] = [v*0.1 for v in range(100)]
    getBins(df,'x',10)
    getBinQuan(df['x'],df['y'])
    
    tarlist = ['x']
    X_trans, propdic = getTransData(df, df['y'], tarlist)
    print(X_trans.head(20))
    
    df_t = pd.DataFrame()
    df_t['x'] = range(1000)
    df_t['y'] = [v*0.1 for v in range(1000)]
    X_test = getTransTest(df_t, propdic)
    print(X_test.head(20))

