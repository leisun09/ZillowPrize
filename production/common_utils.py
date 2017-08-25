# Common utils shared by different models.
from sklearn.preprocessing import LabelEncoder
import pandas as pd

FOLD_NUM = 10

TRAINING_DATA_FILE = "../../data/train_2016_v2.csv"
PROPERTIES_FILE = "../../data/properties_2016.csv"
TEST_DATA_FILE = "../../data/sample_submission.csv"
RESULT_FILE = "../../data/result.csv"


def read_training_data():
    print('Reading training data.')
    return pd.read_csv(TRAINING_DATA_FILE)


def read_properties_data():
    print('Reading properties data.')
    return pd.read_csv(PROPERTIES_FILE)


def read_test_data():
    print('Reading test data.')
    return pd.read_csv(TEST_DATA_FILE)


def encode_data(properties_df):
    print('Encoding missing data.')
    for column in properties_df.columns:
        if properties_df[column].dtype == 'object':
            properties_df[column].fillna(-1, inplace=True)
            label_encoder = LabelEncoder()
            list_value = list(properties_df[column].values)
            label_encoder.fit(list_value)
            properties_df[column] = label_encoder.transform(list_value)
    return properties_df


def get_train_properties_df(train_df, properties_df):
    print('Combining training with properties data.')
    return train_df.merge(properties_df, how='left', on='parcelid')


def get_model_train_valid_data(train_properties_df):
    print('Creating training and validation data.')
    train_index = []
    valid_index = []
    for i in xrange(len(train_properties_df)):
        if i % FOLD_NUM != 0:
            train_index.append(i)
        else:
            valid_index.append(i)
    train_dataset = train_properties_df.iloc[train_index]
    valid_dataset = train_properties_df.iloc[valid_index]
    # create train and valid data.
    train_x = train_dataset.drop(
        ['parcelid', 'logerror', 'transactiondate'], axis=1)
    train_y = train_dataset['logerror'].values
    valid_x = valid_dataset.drop(
        ['parcelid', 'logerror', 'transactiondate'], axis=1)
    valid_y = valid_dataset['logerror'].values
    return train_x, train_y, valid_x, valid_y


def get_test_properties_df(test_df, properties_df):
    print('Combining test with properties data.')
    test_df['parcelid'] = test_df['ParcelId']
    return test_df.merge(properties_df, how='left', on='parcelid')


def predict_test(test_predict):
    print('Predicting on test data.')
    tmp_df = read_test_data()
    for column in tmp_df.columns[tmp_df.columns != 'ParcelId']:
        tmp_df[column] = test_predict
    return tmp_df


def write_result(result_df):
    print('Writing to csv.')
    result_df.to_csv(RESULT_FILE, index=False, float_format='%.4f')
    print('Congratulation!')
