# XGBoost baseline for feature engineering.
#
# Training result: [192] train-mae:0.051412 holdout-mae:0.051941
# Public score: 0.0646266
# {'longitude':21, 'yearbuilt':18, 'taxamount':50}
import common_utils as cu
import xgboost as xgb
import pandas as pd
import numpy as np

class XGBoostModel(object):
    def __init__(self):
        self.base_model = None

    def train(self, X_train, y_train, X_holdout, y_holdout):
        print('Training the model.')
        params = {
            'eta': 0.033,
            'max_depth': 6,
            'subsample': 0.80,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'silent': 1
        }
        xgboost_X_train = xgb.DMatrix(X_train, label=y_train)
        xgboost_X_holdout = xgb.DMatrix(X_holdout, label=y_holdout)
        watchlist = [(xgboost_X_train, 'train'), (xgboost_X_holdout, 'holdout')]
        self.base_model = xgb.train(
            params, xgboost_X_train, 10000, watchlist,
            early_stopping_rounds=100, verbose_eval=10)

    def predict(self, predict_df):
        return self.base_model.predict(xgb.DMatrix(predict_df))


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
    
    def gridSearch():
        st,nt,step=5,51,5
        for a in range(st,nt,step):
            for b in range(st,nt,step):
                rlist = []
                for c in range(st,nt,step):
                    bindic = dict(zip(tarlist, [a, b, c]))
                    X_trans = getTransData(X, tarlist, bindic)
                    # get CV from train data.
                    X_train, y_train, X_holdout, y_holdout = cu.get_cv(X_trans, y)
                
                    # train model.
                    xgbm = XGBoostModel()
                    xgbm.train(X_train, y_train, X_holdout, y_holdout)
                    rlist.append([a, b, c, xgbm.base_model.best_score])
                
                with open('../../data/param.data','a') as outfile:
                    for vs in rlist:
                        outfile.write('\t'.join([str(v) for v in vs]) + '\n')
    
    # read train data.
    X, y = cu.get_train_data(encode_non_object=True)
    tarlist = X.columns
    X_trans, propdic = getTransData(X, tarlist)
    
    for c in tarlist:
        X_trans[c] = X_trans[c].astype(float)
    # get CV from train data.
    X_train, y_train, X_holdout, y_holdout = cu.get_cv(X_trans, y)

    # train model.
    xgbm = XGBoostModel()
    xgbm.train(X_train, y_train, X_holdout, y_holdout)
    
    # read test data.
    T = cu.get_test_data(encode_non_object=True)
    T_trans = getTransTest(T, propdic)
    # predict result.
    print('Predicting.')
    y_pred = xgbm.predict(T_trans[X_train.columns])

    # write result.
    cu.write_result(y_pred)


if __name__ == "__main__":
    run()
