# Linear regression baseline for feature engineering.
#
# Public score: 0.0649163
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import common_utils as cu
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

class LinearRegressionModel(object):
    def __init__(self):
        self.base_model = None

    def train(self, X_train, y_train, X_holdout, y_holdout):
        print('Training model.')
        self.base_model = LinearRegression()
        self.base_model.fit(X_train, y_train)

    def predict(self, predict_df):
        return self.base_model.predict(predict_df)

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
    
    for k in rdic.keys():
        if str(rdic[k]) == 'nan':
            rdic[k] = avg
            
    return rdic

def getTransData(X, y, tarlist):
    X_trans = pd.DataFrame()
    propdic = {}
    for k in X.columns:
        if not k in tarlist:
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

def run():
    # read train data.
    X, y = cu.get_train_data(encode_non_object=True)
    # train model.
    lrm = LinearRegressionModel()
    
    tarlist = [c for c in X.columns if not c in 'fips,hashottuborspa,poolcnt,pooltypeid10,assessmentyear'.split(',')]

    X_trans, propdic = getTransData(X, y, tarlist)
    x_train, y_train, x_holdout, y_holdout = cu.get_cv(X_trans, y)
    lrm.train(x_train, y_train, None, None)
    y_pred = lrm.predict(x_holdout)

    score = abs(y_pred-y_holdout).mean()
    print(score)

    y_trans = [max([min([0.1,v]),-0.1]) for v in y]
    lrm.train(X_trans, y_trans, None, None)

    # read test data.
    T = cu.get_test_data(encode_non_object=True)
    T_trans = getTransTest(T, propdic)
    # predict result.
    print('Predicting.')
    y_pred = lrm.predict(T_trans[X_trans.columns].values)
    
    # write result.
    cu.write_result(y_pred)
    print(max(list(lrm.base_model.coef_)))
    print(min(y_pred))

if __name__ == "__main__":
    run()
