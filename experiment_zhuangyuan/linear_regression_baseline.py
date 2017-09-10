# Linear regression baseline for feature engineering.
#
# Public score: 0.0649163
from sklearn.linear_model import LinearRegression
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


def run():
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
            valnum = X[k].value_counts().shape[0]
            if not k in tarlist:
                X_trans[k] = X[k]
            else:
                try:
                    bins = getBins(X, k, 30)
                    if bins[1] > 0:
                        bins.insert(1, 0.0)
                except:
                    continue
                if valnum > 20:
                    aseri = pd.cut(X[k],bins,include_lowest=True,labels=range(len(bins)-1))    
    
                else:
                    aseri = X[k]
                
                mapdic = getBinQuan(aseri, y)
                X_trans[k] = aseri.apply(lambda x:mapdic[x])
                propdic[k] = propdic.get(k, {})
                propdic[k]['bins'] = bins
                propdic[k]['mapdic'] = mapdic
        
        return X_trans
    # read train data.
    X, y = cu.get_train_data(encode_non_object=True)

    # train model.
    lrm = LinearRegressionModel()
    
    tarlist = X.columns#['longitude', 'yearbuilt', 'taxamount']

    X_trans = getTransData(X, tarlist)
    x_train, y_train, x_holdout, y_holdout = cu.get_cv(X_trans, y)
    lrm.train(x_train, y_train, None, None)
    y_pred = lrm.predict(x_holdout)

    score = abs(y_pred-y_holdout).mean()
    print(score)

    
    return 0
    
    lrm.train(X, y, None, None)

    # read test data.
    T = cu.get_test_data(encode_non_object=True)

    # predict result.
    print('Predicting.')
    y_pred = lrm.predict(T[X.columns])

    # write result.
    cu.write_result(y_pred)


if __name__ == "__main__":
    run()
