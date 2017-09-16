# XGBoost baseline for feature engineering.
#
# Training result: [192] train-mae:0.051412 holdout-mae:0.051941
# Public score: 0.0646266
# {'longitude':21, 'yearbuilt':18, 'taxamount':50}
import common_utils as cu
import xgboost as xgb
import pandas as pd
import numpy as np
import data_trans as dt

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
    def gridSearch():
        st,nt,step=5,51,5
        for a in range(st,nt,step):
            for b in range(st,nt,step):
                rlist = []
                for c in range(st,nt,step):
                    bindic = dict(zip(tarlist, [a, b, c]))
                    X_trans = dt.getTransData(X, tarlist, bindic)
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
    X_trans, propdic = dt.getTransData(X, y, tarlist)
    
    for c in tarlist:
        X_trans[c] = X_trans[c].astype(float)
    # get CV from train data.
    X_train, y_train, X_holdout, y_holdout = cu.get_cv(X_trans, y)

    # train model.
    xgbm = XGBoostModel()
    xgbm.train(X_train, y_train, X_holdout, y_holdout)
    
    # read test data.
    T = cu.get_test_data(encode_non_object=True)
    T_trans = dt.getTransTest(T, propdic)
    # predict result.
    print('Predicting.')
    y_pred = xgbm.predict(T_trans[X_train.columns])

    # write result.
    cu.write_result(y_pred)


if __name__ == "__main__":
    run()
