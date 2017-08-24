# XGBoost baseline for feature engineering. Highlights:
# 1. Encode missing data for object type and leave other types missing data for
#    xgboost to handle it.
# 2. Train data: 2016 Jan. to Sept. + 50% of 2016 Oct. to Dec.
#    Valid data: remaining 50% of 2016 Oct. to Dec.
# 3. Test data in public and private board is 2016 Oct. to Dec. and
#    2017 Oct. to Dec. respectively. That's the reason we split the data.
#
# Training result: [192] train-mae:0.066748 valid-mae:0.065658
# Public score: 0.0655300
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

FIRST_2016_OCT_INDEX = 81760

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
    print('Creating training and validation data for xgboost.')
    train_properties_df =\
        train_df.merge(properties_df, how='left', on='parcelid')
    # split 2016 Oct. to Dec. data to train and valid data.
    train_index = range(FIRST_2016_OCT_INDEX)
    valid_index = []
    for i in xrange(FIRST_2016_OCT_INDEX, len(train_properties_df)):
        if i % 2 != 0:
            train_index.append(i)
        else:
            valid_index.append(i)
    train_dataset = train_properties_df.iloc[train_index]
    valid_dataset = train_properties_df.iloc[valid_index]
    # create train and valid data for xgboost.
    train_x = train_dataset.drop(
        ['parcelid', 'logerror', 'transactiondate'], axis=1)
    train_y = train_dataset['logerror'].values
    valid_x = valid_dataset.drop(
        ['parcelid', 'logerror', 'transactiondate'], axis=1)
    valid_y = valid_dataset['logerror'].values
    return [xgb.DMatrix(train_x, label=train_y),
            xgb.DMatrix(valid_x, label=valid_y),
            train_x.columns]


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


def get_model_test_data():
    print('Building test set.')
    test_df['parcelid'] = test_df['ParcelId']
    test_properties_df = test_df.merge(properties_df, how='left', on='parcelid')
    return xgb.DMatrix(test_properties_df[train_columns])


def predict_test():
    print('Predicting on test data.')
    test_predict = xgboost_model.predict(xgboost_test_df)
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
    xgboost_train_df, xgboost_valid_df, train_columns =\
        get_model_train_valid_data()
    xgboost_model = train_model()
    xgboost_test_df = get_model_test_data()
    result_df = predict_test()
    write_result()

