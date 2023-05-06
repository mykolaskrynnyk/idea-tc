"""
Standardised routines for optimising a scikit-learn model for text classification using grid search.
"""
# standard library
from typing import Optional

# machine learning
from sklearn.model_selection import GridSearchCV, PredefinedSplit


def get_parameters(model) -> dict:
    """
    Extract parameters from a tuned scikit-learn estimator.

    Parameters
    ----------
    model : estimator object
        A tuned scikit-learn estimator object, e.g., a model, pipeline or similar.

    Returns
    -------
    parameters : dict
        Dictionary of optimal hyperparameters that can be passed to a blank estimator object to train a new one
        from scratch.
    """
    parameters = model.get_params()
    return parameters


def search_model(
        model,
        params: dict,
        x_train: list[str],
        y_train: list[int],
        x_valid: Optional[list[str]] = None,
        y_valid: Optional[list[int]] = None,
) -> dict:
    """
    Search for optimal hyperparameters on a given dataset.

    If validation data is not available, it uses a 3-fold cross-validation method. If validation data is available,
    `PredefinedSplit` object is used for cv.

    Parameters
    ----------
    model : estimator object
        A scikit-learn estimator object, e.g., a model, pipeline or similar.
    params : dict
        Dictionary of custom parameters to be passed to the model for search, i.e., parameter grid.
    x_train : list[str]
        List of training texts.
    y_train : list[int]
        List of training labels.
    x_valid : Optional[list[str]], default=None
        List of validation texts if available.
    y_valid : Optional[list[int]], default=None
        List of validation labels if available.

    Returns
    -------
    parameters : dict
        Parameters of the optimal model found.
    """
    if x_valid is not None:
        test_fold = [-1] * len(x_train) + [0] * len(x_valid)
        cv_splits = PredefinedSplit(test_fold=test_fold)
        # `test_fold` is used to tell traing and validation data apart during cv
        x_train = x_train + x_valid
        y_train = y_train + y_valid

    grid = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring='f1_weighted',
        n_jobs=-1,
        refit=True,
        cv=3 if x_valid is None else cv_splits,
        verbose=0,
        error_score=0.,
    )
    grid.fit(x_train, y_train)
    model = grid.best_estimator_
    params = get_parameters(model)
    return params


def train_predict(model, params : dict, x_train: list[str], y_train: list[int], x_test: list[str]) -> list[int]:
    """
    Train a fastText model on the training data and return prediction for test texts.

    This function only returns a single label per text from a softmax distribution. The label predicted with the
    maximum probability is returned and no threshold is applied.

    Parameters
    ----------
    model : estimator object
        A scikit-learn estimator object, e.g., a model, pipeline or similar.
    params : dict
        Dictionary of optimal model parameters.
    x_train : list[str]
        List of training texts.
    y_train : list[int]
        List of training labels.
    x_test : list[str]
        List of test texts for which labels will be predicted.

    Returns
    -------
    y_pred : list[int]
        List of predicted labels.
    """
    model.set_params(**params)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test).tolist()
    return y_pred
