"""
Utilities for working with fastText module in Python.
"""
# standard library
import re
from typing import Iterable

# local packages
from src.utils import dataset_types


def write_dataset(dataset_type: dataset_types, texts: Iterable[str], labels: Iterable[int]) -> None:
    """
    Utility function to write texts and labels to disk in a format suitable for training a fastText model.

    See https://fasttext.cc/docs/en/python-module.html#text-classification-model for details on the format.
    All whitespace characters are replaced with a space as per preprocessing conventions, see
    https://fasttext.cc/docs/en/python-module.html#important-preprocessing-data-encoding-conventions

    Parameters
    ----------
    dataset_type : dataset_types
        One of the two file names where 'data.train' will be used for training and 'data.valid' for validation.
    texts : Iterable[str]
        Iterable of texts. Note that each text must be written in a single line so no linebreaks are allowed in any
        of the texts.
    labels : Iterable[int]
        Iterable of integer labels. While fastText can be used for multilabel training, this function is designed
        for multiclass tasks only.

    Returns
    -------
    None
    """
    with open(dataset_type, 'w') as file:
        for text, label in zip(texts, labels):
            text = re.sub(pattern=r'\s+', repl=' ', string=text)
            file.write(f'__label__{label} {text}\n')
