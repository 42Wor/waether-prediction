import numpy as np
import json


class MyPolynomial_R:
    def __init__(self, degree=2):
        self.degree = degree
        self.coef_ = None
        self.intercept_ = None

    def _poly_features(self, X):
        X = np.array(X)
        n_samples, n_features = X.shape
        from itertools import combinations_with_replacement
        combs = [combinations_with_replacement(range(n_features), d) for d in range(1, self.degree + 1)]
        combs = [item for sublist in combs for item in sublist]
        X_poly = [np.prod(X[:, comb], axis=1) for comb in combs]
        X_poly = np.vstack([np.ones(n_samples)] + X_poly).T
        return X_poly

    def fit(self, X, y):
        X_poly = self._poly_features(X)
        y = np.array(y).flatten()
        theta = np.linalg.pinv(X_poly.T @ X_poly) @ X_poly.T @ y
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]

    def predict(self, X):
        X_poly = self._poly_features(X)
        return X_poly @ np.concatenate(([self.intercept_], self.coef_))

    def score(self, X, y):
        y = np.array(y).flatten()
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

    def save(self, filename):
        model_data = {
            'coef': self.coef_.tolist(),
            'intercept': self.intercept_,
            'degree': self.degree
        }
        with open(filename, 'w') as f:
            json.dump(model_data, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'r') as f:
            model_data = json.load(f)
        model = cls(degree=model_data.get('degree', 2))
        model.coef_ = np.array(model_data['coef'])
        model.intercept_ = model_data['intercept']
        return model


def my_train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True):
    if len(arrays) == 0:
        raise ValueError("At least one array required as input.")

    n_samples = len(arrays[0])
    for array in arrays:
        if len(array) != n_samples:
            raise ValueError("All input arrays must have the same length.")

    # Determine test and train sizes
    if test_size is None and train_size is None:
        test_size = 0.25

    n_test, n_train = None, None

    # Calculate n_test based on test_size
    if test_size is not None:
        if isinstance(test_size, float):
            if not 0 <= test_size <= 1:
                raise ValueError("test_size must be between 0 and 1 when a float.")
            n_test = int(n_samples * test_size)
        elif isinstance(test_size, int):
            if not 0 <= test_size <= n_samples:
                raise ValueError(f"test_size must be between 0 and {n_samples} when an integer.")
            n_test = test_size
        else:
            raise TypeError("test_size must be either int or float.")

    # Calculate n_train based on train_size
    if train_size is not None:
        if isinstance(train_size, float):
            if not 0 <= train_size <= 1:
                raise ValueError("train_size must be between 0 and 1 when a float.")
            n_train = int(n_samples * train_size)
        elif isinstance(train_size, int):
            if not 0 <= train_size <= n_samples:
                raise ValueError(f"train_size must be between 0 and {n_samples} when an integer.")
            n_train = train_size
        else:
            raise TypeError("train_size must be either int or float.")

    # Handle cases where only one of test_size or train_size is specified
    if test_size is not None and train_size is not None:
        if n_test + n_train > n_samples:
            raise ValueError(
                f"The sum of test_size ({n_test}) and train_size ({n_train}) exceeds samples ({n_samples}).")
    else:
        if test_size is not None:
            n_train = n_samples - n_test
        elif train_size is not None:
            n_test = n_samples - n_train

    # Ensure non-negative sizes
    if n_train < 0 or n_test < 0:
        raise ValueError("Train and test sizes must be non-negative.")

    # Generate indices
    if shuffle:
        rng = np.random.RandomState(random_state)
        indices = rng.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    # Split indices
    train_indices = indices[:n_train]
    test_indices = indices[n_train:n_train + n_test]

    # Split each array
    split_arrays = []
    for array in arrays:
        try:
            # Handle pandas DataFrame/Series without converting to numpy
            if hasattr(array, 'iloc'):
                train = array.iloc[train_indices]
                test = array.iloc[test_indices]
            else:
                array = np.asarray(array)
                train = array[train_indices]
                test = array[test_indices]
        except TypeError:
            # Handle lists and other indexable types
            array = np.asarray(array)
            train = array[train_indices]
            test = array[test_indices]
        split_arrays.extend([train, test])

    return tuple(split_arrays)
