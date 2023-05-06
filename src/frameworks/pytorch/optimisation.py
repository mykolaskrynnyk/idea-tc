# standard library
from typing import Optional

# data wrangling
import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# deep learning
import torch

# local packages
from . import training, inference, utils


def get_parameters(model) -> dict:
    parameters = {
        'count': utils.count_parameters(model),
        'epochs': model.metadata['epochs']
    }
    return parameters


def search_model(
        model,
        params: dict,
        x_train: list[str],
        y_train: list[int],
        x_valid: Optional[list[str]] = None,
        y_valid: Optional[list[int]] = None,
):
    if x_valid is None:
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=.25)
    dataset_train = Dataset.from_dict({'tokens': x_train, 'label': y_train}).with_format('torch')
    dataset_valid = Dataset.from_dict({'tokens': x_valid, 'label': y_valid}).with_format('torch')

    batch_size = params.get('batch_size', 32)
    collate_fn = utils.collate_batch_fixed
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_valid = DataLoader(dataset_valid, batch_size=128, shuffle=False, collate_fn=collate_fn)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    epoch = training.train(
        model=model,
        epochs=params.get('epochs', 10),
        optimiser=optimiser,
        criterion=criterion,
        dataloader_train=dataloader_train,
        dataloader_valid=dataloader_valid,
        patience=params.get('patience', 3),
    )
    params = {'epochs': epoch, 'batch_size': batch_size}
    return params


def train_predict(model, params: dict, x_train: list[str], y_train: list[int], x_test: list[str]) -> list[int]:
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
    y_dummy = np.zeros(len(x_test), dtype=np.int32)
    dataset_train = Dataset.from_dict({'tokens': x_train, 'label': y_train}).with_format('torch')
    dataset_test = Dataset.from_dict({'tokens': x_test, 'label': y_dummy}).with_format('torch')

    collate_fn = utils.collate_batch_fixed
    dataloader_train = DataLoader(dataset_train, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset_test, batch_size=128, shuffle=False, collate_fn=collate_fn)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    training.train(
        model=model,
        epochs=params['epochs'],
        optimiser=optimiser,
        criterion=criterion,
        dataloader_train=dataloader_train,
    )
    y_pred = inference.predict(model=model, dataloader=dataloader_test)
    return y_pred
