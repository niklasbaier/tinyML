import os
import sys
import pytest

from sklearn import datasets, model_selection

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


@pytest.fixture
def sample_data_classification():
    sample_dataset = datasets.load_breast_cancer()
    X, y = sample_dataset.data, sample_dataset.target
    return model_selection.train_test_split(X, y, test_size=0.2, random_state=1234)


@pytest.fixture
def sample_data_regression():
    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4
    )
    return model_selection.train_test_split(X, y, test_size=0.2, random_state=1234)
