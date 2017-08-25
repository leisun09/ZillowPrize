# Generate completed training and test data, which is time-consuming to
# generate every time.
import common_utils as cu
import pandas as pd

TRAIN_DATA_FILE = "../../data/train_2016_v2.csv"
PROPERTIES_FILE = "../../data/properties_2016.csv"
TEST_DATA_FILE = "../../data/sample_submission.csv"

COMPLETED_TRAIN_DATA_FILE = "../../data/completed_train.csv"
COMPLETED_TEST_DATA_FILE = "../../data/completed_test.csv"


def get_completed_train_data():
    return pd.read_csv(COMPLETED_TRAIN_DATA_FILE)


def get_completed_test_data():
    return pd.read_csv(COMPLETED_TEST_DATA_FILE)


def get_test_data():
    return pd.read_csv(TEST_DATA_FILE)


def generate_train_data():
    print('Generating train data.')
    train_df = pd.read_csv(TRAIN_DATA_FILE)
    properties_df = cu.encode_data(pd.read_csv(PROPERTIES_FILE))
    train_properties_df = \
        train_df.merge(properties_df, how='left', on='parcelid')
    train_properties_df.to_csv(
        COMPLETED_TRAIN_DATA_FILE, index=False, float_format='%f')


def generate_test_data():
    print('Generating test data.')
    test_df = pd.read_csv(TEST_DATA_FILE)
    test_df['parcelid'] = test_df['ParcelId']
    properties_df = pd.read_csv(PROPERTIES_FILE)
    test_properties_df = test_df.merge(properties_df, how='left', on='parcelid')
    test_properties_df = test_properties_df.drop(
        ['ParcelId', '201610', '201611', '201612', '201710', '201711',
         '201712'], axis=1)
    test_properties_df.to_csv(
        COMPLETED_TEST_DATA_FILE, index=False, float_format='%f')


if __name__ == '__main__':
    generate_train_data()
    generate_test_data()
