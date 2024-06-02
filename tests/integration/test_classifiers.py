import pytest
from sklearn import datasets, model_selection, linear_model

from tinyML import classifiers, commons


@pytest.fixture
def sample_data():
    sample_dataset = datasets.load_breast_cancer()
    X, y = sample_dataset.data, sample_dataset.target
    return model_selection.train_test_split(X, y, test_size=0.2, random_state=1234)


def test_logistic_regression(sample_data):
    X_train, X_test, y_train, y_test = sample_data

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
