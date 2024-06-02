import pytest
from sklearn import datasets, model_selection, linear_model

from tinyML import regressors, commons


@pytest.fixture
def sample_data():
    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4
    )
    return model_selection.train_test_split(X, y, test_size=0.2, random_state=1234)


@pytest.fixture
def sample_data_classification():
    sample_data = datasets.load_breast_cancer()
    X, y = sample_data.data, sample_data.target
    return model_selection.train_test_split(X, y, test_size=0.2, random_state=1234)


def test_linear_regression_gd(sample_data):
    X_train, X_test, y_train, y_test = sample_data

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


def test_linear_regression_ols(sample_data):
    X_train, X_test, y_train, y_test = sample_data

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


def test_logistic_regression(sample_data_classification):
    X_train, X_test, y_train, y_test = sample_data_classification

    # tinyML model
    model = regressors.LogisticRegression(learning_rate=0.1, epochs=2500)
    model.fit(X_train, y_train)
    y_pred = model.predict_binary(X_test)

    # scikit-learn model
    sklearn_model = linear_model.LogisticRegression(
        penalty=None, solver="sag"
    )  # avoid adding regularization
    sklearn_model.fit(X_train, y_train)
    sklearn_y_pred = sklearn_model.predict(X_test)

    assert commons.accuracy(y_test, y_pred) == pytest.approx(
        commons.accuracy(y_test, sklearn_y_pred)
    )
