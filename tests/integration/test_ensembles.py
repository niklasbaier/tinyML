import pytest
from sklearn import tree

from tinyML import ensembles, commons


def test_decision_tree(sample_data_classification):
    X_train, X_test, y_train, y_test = sample_data_classification

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
