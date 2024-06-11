import pytest
from sklearn import datasets, model_selection, tree

from tinyML import ensembles, commons


@pytest.fixture
def sample_data():
    sample_dataset = datasets.load_breast_cancer()
    X, y = sample_dataset.data, sample_dataset.target
    return model_selection.train_test_split(X, y, test_size=0.2, random_state=1234)


def test_decision_tree(sample_data):
    X_train, X_test, y_train, y_test = sample_data

    # tinyML model
    model = ensembles.DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # scikit-learn model
    sklearn_model = tree.DecisionTreeClassifier(
        criterion="entropy", max_depth=100, min_samples_split=2
    )
    sklearn_model.fit(X_train, y_train)
    sklearn_y_pred = sklearn_model.predict(X_test)

    assert commons.accuracy(y_test, y_pred) == pytest.approx(
        commons.accuracy(y_test, sklearn_y_pred)
    )
