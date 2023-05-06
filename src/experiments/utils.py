"""
Miscellaneous routines for reading, slicing and dicing the data.
"""
# standard library
import json
from pathlib import Path
from math import ceil
from collections import defaultdict
from typing import Optional, Any

# data wrangling
import numpy as np
import pandas as pd
from datasets import Dataset

# local packages
from ..utils import feature_types, metric_types


def get_sample_sizes(dataset: Dataset, max_size: Optional[int] = None) -> list[int]:
    """
    Create sample sizes of power of 2, starting from 256, e.g., [256, ..., 8192, 16384, dataset_size].

    Parameters
    ----------
    dataset : Dataset
        Training set from a Hugging Face Dataset.
    max_size : Optional[int]
        Only create sample sizes below max_size.

    Returns
    -------
    sample_sizes : list[int]
        List of sample sizes.
    """
    n = dataset.num_rows
    power_max = ceil(np.log2(n))
    sample_sizes = [min(2**power, n) for power in range(8, power_max + 1)]
    if max_size is not None:
        sample_sizes = [size for size in sample_sizes if size < max_size]
    return sample_sizes


def get_number_runs(sample_size: int) -> int:
    """
    Get the number of validation runs to perform based on the size of a sample.

    For smaller samples, 10 runs are performed. For medium-sized samples, 3 runs are performed. Only 1 run is used for
    large samples.

    Parameters
    ----------
    sample_size : int
        Sample size used for each run.

    Returns
    -------
    n_runs : int
        Number of validation runs to be performed for a given sample size.
    """
    # define the number of validation runs
    if sample_size <= 2 ** 14:
        n_runs = 10
    elif sample_size <= 2 ** 18:
        n_runs = 3
    else:
        n_runs = 1
    return n_runs


def sample_train_split(dataset: Dataset, feature: feature_types, sample_size: int) -> tuple[list[str], list[int]]:
    """
    Sample `sample_size` examples from the training set.

    Parameters
    ----------
    dataset : Dataset
        Training set from a Hugging Face dataset.
    feature : feature_types
        The name of a text feature to be used for the experiment. 'text_clean' corresponds to preprocessed lemmatised
        texts with ngrams and without punctuation. 'text' corresponds a raw unprocessed texts.
    sample_size : int
        Number of examples to sample from the dataset.

    Returns
    -------
    x_train, y_train : tuple[list[str], list[int]]
        A tuple of training texts and labels.
    """
    sample_indices = np.random.choice(range(dataset.num_rows), replace=False, size=sample_size)
    dataset_sample = dataset.select(sample_indices)
    x_train, y_train = dataset_sample[feature], dataset_sample['label']
    return x_train, y_train


def get_valid_split(dataset: Optional[Dataset], feature: feature_types) -> tuple[Optional[list[str]], Optional[list[int]]]:
    """
    Get validation texts and labels if the dataset has a validation split.

    Parameters
    ----------
    dataset : Optional[Dataset]
        Validation set from a Hugging Face dataset.
    feature : feature_types
        The name of a text feature to be used for the experiment. 'text_clean' corresponds to preprocessed lemmatised
        texts with ngrams and without punctuation. 'text' corresponds a raw unprocessed texts.

    Returns
    -------
    x_valid, y_valid : tuple[Optional[list[str]], Optional[list[int]]]
        A tuple of validation texts and labels if the validation set exists, otherwise, a tuple of two None values.
    """
    if dataset is None:
        x_valid, y_valid = None, None
    else:
        x_valid, y_valid = dataset[feature], dataset['label']
    return x_valid, y_valid


def read_metrics(path_experiments: Path, dataset_name: str) -> pd.DataFrame:
    """
    Read metrics of all models for a given dataset into a single DataFrame.

    Parameters
    ----------
    path_experiments : Path
        Pathlib Path object to the directory sotring '*_metrics.csv' files.
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    df_metrics : pd.DataFrame
        Concatenated DataFrame of metrics for all models.
    """
    paths = path_experiments.glob(f'{dataset_name}*_metrics.csv')
    df_metrics = pd.concat([pd.read_csv(path) for path in paths], axis=0, ignore_index=True)
    df_metrics['parameters'] = df_metrics['parameters'].apply(json.loads)
    return df_metrics


def read_predictions(path_experiments: Path, dataset_name: str) -> pd.DataFrame:
    """
    Read predictions of all models for a text set of a given dataset into a single DataFrame.

    Parameters
    ----------
    path_experiments : Path
        Pathlib Path object to the directory sotring '*_predictions.csv' files.
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    df_preds : pd.DataFrame
        Concatenated DataFrame of predictions made by every model.
    """
    paths = path_experiments.glob(f'{dataset_name}*_predictions.csv')
    df_preds = pd.concat([pd.read_csv(path) for path in paths], axis=1)
    df_preds = df_preds.loc[:, ~df_preds.columns.duplicated()]  # remove repeated 'y_pred' columns
    return df_preds


def show_best_results(
        path_experiments: Path,
        metric_name: metric_types = 'fscore',
        precision: int = 2,
) -> pd.DataFrame:
    """
    Display the highest score for a given metric for each model and dataset.

    Parameters
    ----------
    path_experiments : Path
        Pathlib Path object to the directory sotring '*_metrics.csv' files.
    metric_name : metric_types, default='fscore'
        One of the three available metric names.
    precision : int, default=2
        Number of decimal points to round to.
    Returns
    -------

    """
    results = defaultdict(dict)
    for path in path_experiments.glob('*metrics*'):
        dataset_name, classifier_name = path.name.removesuffix('_metrics.csv').split('_processed_')
        df_metrics = pd.read_csv(path)
        train_size = df_metrics['train_size'].max()
        score = df_metrics.query('target == "total_weighted" and train_size == @train_size')[metric_name].item()
        # score = df_metrics.query('target == @target')[metric_name].max()
        results[dataset_name][classifier_name] = score
    index_order = [
        'dummy_classifier',
        # sparce representations models
        'complement_naive_bayes', 'sgd_classifier',
        # static representations models
        'fasttext', 'cnn',
        # contextual representations models
        'deberta_v3_small_finetuned', 'deberta_v3_small_zeroshot',
        'deberta_v3_xsmall_zeroshot', 'deberta_v3_base_zeroshot',
    ]
    column_order = [
        # sentiment analysis
        'rotten_tomatoes', 'imdb', 'yelp_polarity', 'yelp_review_full', 'setfit_sst5', 'dynabench_dynasent',
        # news categorisation
        'ag_news', '20_newsgroups',
        # topic classification
        'dbpedia_14', 'web_of_science',
    ]
    df_results = pd.DataFrame(results)
    df_results = df_results.reindex(index_order, axis=0).reindex(column_order, axis=1)
    df_results = df_results.multiply(100).round(precision)
    return df_results


def prepare_zsc_batch(
        tokeniser: Any,
        text: str,
        candidate_labels: list[str],
        hypothesis_template: str = 'This example is {}.',
        device: Any = 'cpu',
) -> dict:
    """
    Prepare a batch of examples for zero-shot classification.

    The function constructs a batch from a signle text and a list of candidate labels.

    Parameters
    ----------
    tokeniser : Any
        Pre-trained tokeniser from Hugging Face.
    text : str
        Text to be labelled.
    candidate_labels : list[str]
        List of candidate labels.
    hypothesis_template : str
        Template to be used to constuct the hypothesis for the NLI-style task.
    device : Any
        Name of the device to place the batch to.

    Returns
    -------
    batch : dict
        Batch of encoded examples.
    """
    examples = list()
    for candidate_label in candidate_labels:
        examples.append([text, hypothesis_template.format(candidate_label)])
    # use 'only_first' to avoid truncating the hypothesis
    batch = tokeniser(examples,  padding=True, truncation='only_first', return_tensors='pt')
    if device != 'cpu':
        for k, v in batch.items():
            batch[k] = v.to(device)
    return batch
