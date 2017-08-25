# Linear regression baseline for feature engineering.
#
# Public score: 0.0651228
from sklearn.linear_model import LinearRegression
import common_utils as cu


if __name__ == "__main__":
    # read train data.
    X, y = cu.get_train_data(encode_non_object=True)

    # train model.
    print('Training model.')
    model = LinearRegression()
    model.fit(X, y)

    # read test data.
    T = cu.get_test_data(encode_non_object=True)

    # predict result.
    print('Predicting.')
    y_pred = model.predict(T[X.columns])

    # write result.
    cu.write_result(y_pred)
