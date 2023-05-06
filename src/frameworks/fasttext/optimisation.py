"""
Standardised routines for optimising and using a fastText model for text classification.
"""
# standard library
import os
import re
from math import log2
from typing import Optional

# machine learning
from sklearn.model_selection import train_test_split
from fasttext.FastText import _FastText

# local packages
from .utils import write_dataset


def get_parameters(model: _FastText) -> dict:
    """
    Extract parameters from an auto-tuned fastText model.

    This is based on https://github.com/facebookresearch/fastText/issues/887#issuecomment-649018188
    and documentation at https://fasttext.cc/docs/en/python-module.html#train_supervised-parameters

    Parameters
    ----------
    model : _FastText
        A fine-tuned fastText model object with optimal hyperparameters.

    Returns
    -------
    parameters : dict
        Dictionary of optimal hyperparameters that can be passed to a blank fastText model object to train a new one
        from scratch.
    """
    parameters = dict()
    valid_parameters = {
        'input', 'lr', 'dim', 'ws', 'epoch', 'minCount', 'minCountLabel', 'minn', 'maxn', 'neg', 'wordNgrams', 'loss',
        'bucket', 'thread', 'lrUpdateRate', 't', 'label', 'verbose', 'pretrainedVectors',
    }
    args_obj = model.f.getArgs()
    for parameter in dir(args_obj):
        if parameter in valid_parameters:
            parameters[parameter] = getattr(args_obj, parameter)
    parameters['loss'] = str(parameters.get('loss', 'loss.softmax')).split('.')[-1]
    parameters.pop('input', None)
    return parameters


def search_model(
        model: _FastText,
        params: dict,
        x_train: list[str],
        y_train: list[int],
        x_valid: Optional[list[str]] = None,
        y_valid: Optional[list[int]] = None,
) -> dict:
    """
    Search for optimal hyperparameters on a given dataset.

    If validation data is not available, a 30% split of the training data is used for validation.
    This function uses Automatic hyperparameter optimization from the package authors. For details, see
    https://fasttext.cc/docs/en/autotune.html

    Parameters
    ----------
    model : _FastText
        A blank fastText model object to be used for model
    params : dict
        Dictionary of custom parameters to be passed to the model for training.
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
    params : dict
        Parameters of the optimal model found.
    """
    autotune_duration = 15 * int(log2(len(x_train)) - log2(256) + 1)  # in seconds
    if x_valid is None:
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=.25, stratify=y_train)
    write_dataset(dataset_type='data.train', texts=x_train, labels=y_train)
    write_dataset(dataset_type='data.valid', texts=x_valid, labels=y_valid)
    model = model.train_supervised(
        input='data.train',
        autotuneValidationFile='data.valid',
        autotuneDuration=autotune_duration,
        verbose=0,
        **params,
    )

    # clean up data files
    os.remove('data.train')
    os.remove('data.valid')
    params = get_parameters(model)
    return params


def train_predict(model: _FastText, params: dict, x_train: list[str], y_train: list[int], x_test: list[str]) -> list[int]:
    """
    Train a fastText model on the training data and return prediction for test texts.

    This function only returns a single label per text from a softmax distribution. The label predicted with the
    maximum probability is returned and no threshold is applied.

    Parameters
    ----------
    model : _FastText
        A blank fastText model object to be trained.
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
    write_dataset(dataset_type='data.train', texts=x_train, labels=y_train)
    model = model.train_supervised(input='data.train', **params)
    x_test = [re.sub(pattern=r'\s+', repl=' ', string=x) for x in x_test]
    y_pred = [int(x[0].replace('__label__', '')) for x in model.predict(x_test)[0]]
    os.remove('data.train')  # clean up
    return y_pred
