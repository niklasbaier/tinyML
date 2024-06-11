import collections
import numpy as np

from tinyML import commons


class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # value will only be passed to leaf nodes

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(
        self, min_samples_split: int = 2, max_depth: int = 100, n_features=None
    ):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y_true):
        self.n_features = (
            X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        )
        self.root = self._grow_tree(X, y_true)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y_true, depth: int = 0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y_true))

        # check stopping criteria
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = self._get_most_common_label(y_true)
            return Node(value=leaf_value)

        # find best split
        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)
        best_feature, best_threshold = self._get_best_split(X, y_true, feat_idxs)

        # create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y_true[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y_true[right_idxs], depth + 1)
        return Node(best_feature, best_threshold, left, right)

    def _get_most_common_label(self, y_true):
        counter = collections.Counter(y_true)
        return counter.most_common(1)[0][0]

    def _get_best_split(self, X, y_true, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._get_information_gain(y_true, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx, split_threshold = feat_idx, threshold

        return split_idx, split_threshold

    def _get_information_gain(self, y_true, X_column, threshold):
        parent_entropy = commons.entropy(y_true)

        left_idxs, right_idxs = self._split(X_column, threshold)
        if not len(left_idxs) or not len(right_idxs):
            return 0

        n = len(y_true)
        n_left, n_right = len(left_idxs), len(right_idxs)
        entropy_left, entropy_right = (
            commons.entropy(y_true[left_idxs]),
            commons.entropy(y_true[right_idxs]),
        )
        child_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right

        return commons.information_gain(parent_entropy, child_entropy)

    def _split(self, X_column, threshold):
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column >= threshold).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


# TODO:
class DecisionTreeRegressor:
    pass
