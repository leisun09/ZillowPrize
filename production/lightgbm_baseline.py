# LightGBM baseline for feature engineering.
#
# Training result: [719] train's mae: 0.0675895 valid's mae: 0.0644252
# Public score: 0.0647086
import common_utils as cu
import lightgbm as lgbm


def train_model():
    print('Training the model.')
    params = {
        'max_bin': 10,
        'learning_rate': 0.0021,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mae',
        'sub_feature': 0.5,
        'bagging_fraction': 0.85,
        'bagging_freq': 40,
        'num_leaves': 512,
        'min_data': 500,
        'min_hessian': 0.05,
        'verbose': 0
    }
    return lgbm.train(params, lgbm_train_df, num_boost_round=1000,
                      valid_sets=[lgbm_train_df, lgbm_valid_df],
                      valid_names=['train', 'valid'],
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
    train_x, train_y, valid_x, valid_y = \
        cu.get_model_train_valid_data(train_properties_df)
    lgbm_train_df = lgbm.Dataset(train_x, label=train_y)
    lgbm_valid_df = lgbm.Dataset(valid_x, label=valid_y)
    lgbm_test_df = test_properties_df[train_x.columns]
    # predict result.
    lgbm_model = train_model()
    test_predict = lgbm_model.predict(lgbm_test_df)
    # write result.
    result_df = cu.predict_test(test_predict)
    cu.write_result(result_df)
