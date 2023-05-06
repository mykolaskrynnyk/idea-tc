"""
Routines allowing to experiment with scikit-learn and fastText models for supervised training and text classification.
"""
# standard library
import json
from typing import Callable, Any

# data wrangling
import numpy as np
import pandas as pd
from datasets import DatasetDict

# utils
from tqdm import tqdm

# local packages
from . import evaluation, utils
from ..utils import feature_types


def run_experiment(
        dataset_dict: DatasetDict,
        feature: feature_types,
        get_model: Callable,
        search_params: dict,
        optimisation: Any,
        sample_sizes: list[int],
        max_runs: int = 10,
        progress_bar: bool = True,
        experiment_id: str = '',
        save_path: str = '',
) -> bool:
    """
    Run a series of experiments using "classic" approach, i.e., models from scikit-learn and fastText.

    Parameters
    ----------
    dataset_dict : DatasetDict
        Dataset dictionary object with a 'train', 'test' and (optionally) 'validation' splits.
    feature : feature_types
        The name of a text feature to be used for the experiment. 'text_clean' corresponds to preprocessed lemmatised
        texts with ngrams and without punctuation. 'text' corresponds a raw unprocessed texts.
    get_model : Callable
        Function that returns a blank model object to be trained.
    search_params : dict
        Custom parameters to be passed to `search_model_func`.
    optimisation : Any
        Object that implements `search_model`, `train_predict` and `get_parameters` functions.
    sample_sizes : list[int]
        List of sample sizes. Each sample size is used to optimise and score a model.
    max_runs : int
        Limit the maximum number of experiments for all sample sizes to max_runs.
    progress_bar : bool, default=True
        If True, use a tqdm progress bar over the sample sizes.
    experiment_id : str, default=''
        A string ID to be assigned to all metrics of the experiment.
    save_path : str, default=''
        If not None, it must be a path including a filename prefix (but no extension) to save experiments and
        predictions to.

    Returns
    -------
    True after the experiment results have been written to disk.
    """
    target_names = dataset_dict['train'].features['label'].names
    x_test, y_test = dataset_dict['test'][feature], dataset_dict['test']['label']
    df_list = list()

    with tqdm(sample_sizes, disable=not progress_bar) as t:
        for sample_size in t:
            # conduct hyperparameter tuning for each sample size
            x_train, y_train = utils.sample_train_split(dataset_dict['train'], feature=feature, sample_size=sample_size)
            x_valid, y_valid = utils.get_valid_split(dataset_dict.get('validation'), feature=feature)
            model = get_model()
            t.set_description(f'Training size: {sample_size:,} Running grid search...')
            params = optimisation.search_model(model, search_params, x_train, y_train, x_valid, y_valid,)
            metadata = {
                'experiment_id': experiment_id,
                'train_size': sample_size,
                'parameters': json.dumps(params, default=str),
            }

            # for each sample size, repeat training to get a variance estimate
            t.set_description(f'Training size: {sample_size:,} Starting runs...')
            n_runs = utils.get_number_runs(sample_size=sample_size)
            n_runs = min(max_runs, n_runs)
            for n in range(n_runs):
                x_train, y_train = utils.sample_train_split(dataset_dict['train'], feature=feature, sample_size=sample_size)
                x_valid, y_valid = utils.get_valid_split(dataset_dict.get('validation'), feature=feature)  # only used in DeBERTa
                model = get_model()
                y_pred = optimisation.train_predict(model, params, x_train, y_train, x_test, x_valid, y_valid)
                metadata['run_number'] = n
                df_metrics = evaluation.calculate_run_metrics(y_test, y_pred, target_names, metadata)
                f1_score = df_metrics.query('target == "total_weighted"')['fscore'].item()
                df_list.append(df_metrics)
                t.set_description(f'Training size: {sample_size:,} Run: {n + 1}/{n_runs} F1-score: {f1_score:.3f}')

    df_predictions = pd.DataFrame({'y_true': y_test, experiment_id: y_pred})
    df_metrics = evaluation.aggregate_run_metrics(df_list=df_list)

    # save metrics and predicted labels to disk
    df_metrics.to_csv(f'{save_path}_{experiment_id}_metrics.csv', index=False)
    df_predictions.to_csv(f'{save_path}_{experiment_id}_predictions.csv', index=False)

    return True
