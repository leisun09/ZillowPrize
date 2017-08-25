# XGBoost baseline for feature engineering.
#
# Training result: [189] train-mae:0.066996 valid-mae:0.065312
# Public score: 0.0656603
import common_utils as cu
import xgboost as xgb


def train_model():
    print('Training the model.')
    params = {
        'eta': 0.033,
        'max_depth': 6,
        'subsample': 0.80,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'silent': 1
    }
    watchlist = [(xgboost_X_train, 'train'), (xgboost_X_holdout, 'holdout')]
    return xgb.train(params, xgboost_X_train, 10000, watchlist,
                     early_stopping_rounds=100, verbose_eval=10)


if __name__ == "__main__":
    # read train data.
    X, y = cu.get_train_data(encode_non_object=False)

    # get CV from train data.
    X_train, y_train, X_holdout, y_holdout = cu.get_cv(X, y)

    # train model.
    print('Training model.')
    xgboost_X_train = xgb.DMatrix(X_train, label=y_train)
    xgboost_X_holdout = xgb.DMatrix(X_holdout, label=y_holdout)
    xgboost_model = train_model()

    # read and prepare test data.
    T = cu.get_test_data(encode_non_object=False)
    xgboost_T = xgb.DMatrix(T[X_train.columns])

    # predict result.
    print('Predicting.')
    y_pred = xgboost_model.predict(xgboost_T)

    # write result.
    cu.write_result(y_pred)
