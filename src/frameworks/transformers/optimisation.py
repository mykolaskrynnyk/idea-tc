# standard library
from typing import Optional

# data wrangling
import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# deep learning
from transformers import AutoTokenizer

# local packages
from . import training, inference


def search_model(
        model,
        params: dict,
        x_train: list[str],
        y_train: list[int],
        x_valid: Optional[list[str]] = None,
        y_valid: Optional[list[int]] = None,
):
    return params


def train_predict(model, params: dict, x_train: list[str], y_train: list[int], x_test: list[str], x_valid, y_valid) -> list[int]:
    """
    Train a CNN model on the training data and return prediction for test texts.

    Parameters
    ----------
    model : PyTorch model
        A blank PyTorch model object to be trained.
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
    tokeniser = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base', use_fast=False)
    tokenise = lambda example: tokeniser(example['text'], padding='max_length', truncation=True, max_length=512)
    dataset_train = Dataset.from_dict({'text': x_train, 'label': y_train}).map(tokenise).with_format('torch')
    if x_valid is None:
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=.25)
    dataset_valid = Dataset.from_dict({'text': x_valid, 'label': y_valid}).map(tokenise).with_format('torch')
    dataset_test = Dataset.from_dict({'text': x_test}).map(tokenise).with_format('torch')
    trainer = training.train(
        model=model,
        dataset_train=dataset_train,
        dataset_valid=dataset_valid,
        batch_size=params.get('batch_size', 32),
        patience=params.get('patience', 3),
    )
    y_pred = inference.predict(model=trainer, dataset=dataset_test)
    return y_pred
