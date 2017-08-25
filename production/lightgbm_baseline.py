# LightGBM baseline for feature engineering.
#
# Training result: [719] train's mae: 0.0675899 valid's mae: 0.0644243
# Public score: 0.0647060
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
    return lgbm.train(params, lgbm_X_train, num_boost_round=1000,
                      valid_sets=[lgbm_X_train, lgbm_X_holdout],
                      valid_names=['train', 'holdout'],
                      early_stopping_rounds=100, verbose_eval=10)


if __name__ == "__main__":
    # read train data.
    X, y = cu.get_train_data()

    # get CV from train data.
    X_train, y_train, X_holdout, y_holdout = cu.get_cv(X, y)

    # train model.
    lgbm_X_train = lgbm.Dataset(X_train, label=y_train)
    lgbm_X_holdout = lgbm.Dataset(X_holdout, label=y_holdout)
    lgbm_model = train_model()

    # read and prepare test data.
    T = cu.get_test_data()
    lgbm_T = T[X_train.columns]

    # predict result.
    y_pred = lgbm_model.predict(lgbm_T)

    # write result.
    cu.write_result(y_pred)
