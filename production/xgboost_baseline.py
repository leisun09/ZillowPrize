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
    watchlist = [(xgboost_train_df, 'train'), (xgboost_valid_df, 'valid')]
    return xgb.train(params, xgboost_train_df, 10000, watchlist,
                     early_stopping_rounds=100, verbose_eval=10)


if __name__ == "__main__":
    # read data.
    train_df = cu.read_training_data()
    properties_df = cu.read_properties_data()
    test_df = cu.read_test_data()
    properties_df = cu.encode_data(properties_df)
    # combine data with properties.
    train_properties_df = cu.get_train_properties_df(train_df, properties_df)
    test_properties_df = cu.get_test_properties_df(test_df, properties_df)
    # get train, valid and test data for model.
    train_x, train_y, valid_x, valid_y =\
        cu.get_model_train_valid_data(train_properties_df)
    xgboost_train_df = xgb.DMatrix(train_x, label=train_y)
    xgboost_valid_df = xgb.DMatrix(valid_x, label=valid_y)
    xgboost_test_df = xgb.DMatrix(test_properties_df[train_x.columns])
    # predict result.
    xgboost_model = train_model()
    test_predict = xgboost_model.predict(xgboost_test_df)
    # write result.
    result_df = cu.predict_test(test_predict)
    cu.write_result(result_df)
