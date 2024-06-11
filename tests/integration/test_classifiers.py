import pytest
from sklearn import linear_model

from tinyML import classifiers, commons


def test_logistic_regression(sample_data_classification):
    X_train, X_test, y_train, y_test = sample_data_classification

    # tinyML model
    model = classifiers.LogisticRegression(learning_rate=0.1, epochs=2500)
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
