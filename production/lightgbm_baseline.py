# LightGBM baseline for feature engineering. Highlights:
# 1. Encode missing data for object type and leave other types missing data for
#    lightgbm to handle it.
# 2. Make 10% of training data as validating data.
#
# Training result: [719] train's mae: 0.0675895 valid's mae: 0.0644252
# Public score: 0.0647086
import lightgbm as lgbm
import pandas as pd
from sklearn.preprocessing import LabelEncoder

TRAINING_DATA_FILE = "../../data/train_2016_v2.csv"
PROPERTIES_FILE = "../../data/properties_2016.csv"
TEST_DATA_FILE = "../../data/sample_submission.csv"
RESULT_FILE = "../../data/result.csv"


def read_data():
    print('Reading training data, properties and test data.')
    return [pd.read_csv(TRAINING_DATA_FILE), pd.read_csv(PROPERTIES_FILE),
            pd.read_csv(TEST_DATA_FILE)]


def encode_data():
    print('Encoding missing data.')
    for column in properties_df.columns:
        if properties_df[column].dtype == 'object':
            properties_df[column].fillna(-1, inplace=True)
            label_encoder = LabelEncoder()
            list_value = list(properties_df[column].values)
            label_encoder.fit(list_value)
            properties_df[column] = label_encoder.transform(list_value)


def get_model_train_valid_data():
    print('Creating training and validation data for lightgbm.')
    train_properties_df =\
        train_df.merge(properties_df, how='left', on='parcelid')
    train_index = []
    valid_index = []
    for i in xrange(len(train_properties_df)):
        if i % 10 != 0:
            train_index.append(i)
        else:
            valid_index.append(i)
    train_dataset = train_properties_df.iloc[train_index]
    valid_dataset = train_properties_df.iloc[valid_index]
    # create train and valid data for lightgbm.
    train_x = train_dataset.drop(
        ['parcelid', 'logerror', 'transactiondate'], axis=1)
    train_y = train_dataset['logerror'].values
    valid_x = valid_dataset.drop(
        ['parcelid', 'logerror', 'transactiondate'], axis=1)
    valid_y = valid_dataset['logerror'].values
    return [lgbm.Dataset(train_x, label=train_y),
            lgbm.Dataset(valid_x, label=valid_y),
            train_x.columns]


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


def get_model_test_data():
    print('Building test set.')
    test_df['parcelid'] = test_df['ParcelId']
    test_properties_df = test_df.merge(properties_df, how='left', on='parcelid')
    return test_properties_df[train_columns]


def predict_test():
    print('Predicting on test data.')
    test_predict = lgbm_model.predict(lgbm_test_df)
    tmp_df = pd.read_csv(TEST_DATA_FILE)
    for column in tmp_df.columns[tmp_df.columns != 'ParcelId']:
        tmp_df[column] = test_predict
    return tmp_df


def write_result():
    print('Writing to csv.')
    result_df.to_csv(RESULT_FILE, index=False, float_format='%.4f')
    print('Congratulation!')


if __name__ == "__main__":
    train_df, properties_df, test_df = read_data()
    encode_data()
    lgbm_train_df, lgbm_valid_df, train_columns = get_model_train_valid_data()
    lgbm_model = train_model()
    lgbm_test_df = get_model_test_data()
    result_df = predict_test()
    write_result()
