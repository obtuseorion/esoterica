import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class BaseGramSchmidtForest(BaseEstimator):
    def __init__(self, n_estimators_range=range(10, 201, 10), max_iterations=100, 
                 max_depth=None, min_samples_split=2, random_state=None):
        self.n_estimators_range = n_estimators_range
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.best_n_estimators_, self.best_score_ = self._gram_schmidt_walk(X, y)
        self.trees_ = self._train_trees(X, y, self.best_n_estimators_)
        return self

    def predict(self, X):
        check_is_fitted(self, ['X_', 'y_', 'trees_'])
        X = check_array(X)
        predictions = np.array([tree.predict(X) for tree in self.trees_])
        return self._aggregate_predictions(predictions)

    def _train_trees(self, X, y, n_estimators):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        trees = []
        for _ in range(n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap, y_bootstrap = X[indices], y[indices]
            tree = self._get_tree_model()
            tree.fit(X_bootstrap, y_bootstrap)
            trees.append(tree)
        return trees

    def _gram_schmidt_walk(self, X, y):
        best_n_estimators = 0
        best_score = -np.inf if self._is_regressor() else 0
        for n_estimators in self.n_estimators_range:
            trees = self._train_trees(X, y, n_estimators)
            vectors = [tree.predict(X) for tree in trees]
            vectors = [v / np.linalg.norm(v) for v in vectors]
            x = np.zeros(n_estimators)
            for _ in range(self.max_iterations):
                for t in range(n_estimators):
                    v_perp = vectors[t] - np.sum([x[i] * vectors[i] for i in range(n_estimators) if i != t], axis=0)
                    v_perp /= np.linalg.norm(v_perp)
                    u = np.zeros(n_estimators)
                    u[t] = 1
                    for i in range(n_estimators):
                        if i != t:
                            u[i] = np.dot(v_perp, vectors[i])
                    delta_pos = min(1 - x[i] for i in range(n_estimators) if u[i] > 0)
                    delta_neg = max(-1 - x[i] for i in range(n_estimators) if u[i] < 0)
                    delta = delta_pos if np.random.rand() < delta_pos / (delta_pos - delta_neg) else delta_neg
                    x += delta * u
                selected_trees = [trees[i] for i in range(n_estimators) if x[i] > 0]
                predictions = np.array([tree.predict(X) for tree in selected_trees])
                y_pred = self._aggregate_predictions(predictions)
                score = self._compute_score(y, y_pred)
                if score > best_score:
                    best_score = score
                    best_n_estimators = len(selected_trees)
        return best_n_estimators, best_score

    def _get_tree_model(self):
        raise NotImplementedError("This method should be implemented by derived classes.")

    def _aggregate_predictions(self, predictions):
        raise NotImplementedError("This method should be implemented by derived classes.")

    def _compute_score(self, y_true, y_pred):
        raise NotImplementedError("This method should be implemented by derived classes.")

    def _is_regressor(self):
        raise NotImplementedError("This method should be implemented by derived classes.")

class GramSchmidtForestClassifier(BaseGramSchmidtForest, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        return super().fit(X, y)

    def _get_tree_model(self):
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state
        )

    def _aggregate_predictions(self, predictions):
        return np.array([np.bincount(predictions[:, i]).argmax() for i in range(predictions.shape[1])])

    def _compute_score(self, y_true, y_pred):
        return np.mean(y_pred == y_true)

    def _is_regressor(self):
        return False

class GramSchmidtForestRegressor(BaseGramSchmidtForest, RegressorMixin):
    def _get_tree_model(self):
        return DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state
        )

    def _aggregate_predictions(self, predictions):
        return np.mean(predictions, axis=0)

    def _compute_score(self, y_true, y_pred):
        return -np.mean((y_true - y_pred) ** 2)

    def _is_regressor(self):
        return True
