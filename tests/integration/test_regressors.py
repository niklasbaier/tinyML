import pytest
from sklearn import linear_model

from tinyML import regressors, commons


def test_linear_regression_gd(sample_data_regression):
    X_train, X_test, y_train, y_test = sample_data_regression

    # tinyML model
    model = regressors.LinearRegressionGD(epochs=2000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # scikit-learn model
    sklearn_model = linear_model.LinearRegression()
    sklearn_model.fit(X_train, y_train)
    sklearn_y_pred = sklearn_model.predict(X_test)

    assert commons.mse(y_test, y_pred) == pytest.approx(
        commons.mse(y_test, sklearn_y_pred)
    )  # approx due to gradient descent


def test_linear_regression_ols(sample_data_regression):
    X_train, X_test, y_train, y_test = sample_data_regression

    # tinyML model
    model = regressors.LinearRegressionOLS()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # scikit-learn model
    sklearn_model = linear_model.LinearRegression()
    sklearn_model.fit(X_train, y_train)
    sklearn_y_pred = sklearn_model.predict(X_test)

    assert commons.mse(y_test, y_pred) == pytest.approx(
        commons.mse(y_test, sklearn_y_pred)
    )
