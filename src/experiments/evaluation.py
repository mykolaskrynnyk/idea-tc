"""
Evaluation routines for classifiers.
"""
# data wrangling
import pandas as pd

# evaluation
from sklearn.metrics import precision_recall_fscore_support


def calculate_run_metrics(
    y_test: list[int],
        y_pred: list[int],
        target_names: list[str],
        metadata: dict,
) -> pd.DataFrame:
    """
    Calculate precision, recall and F-score given predictions and ground-truth labels for a binary or multiclass
    classification.

    Parameters
    ----------
    y_test : list[int]
        List of actual ground-truth classes.
    y_pred : list[int]
        List of predicted classes.
    target_names : list[str]
        List of class names that can be used to turn integer indices to string labels.
    metadata : dict
        Arbitrary metadata to be added to the metrics.

    Returns
    -------
    df_metrics : pd.DataFrame
        Pandas DataFrame with len(target_names) + 1 rows where each row corresponds to metrics for a single class
        (or total weighted average) and columns are 'target', 'precision', 'recall', 'fscore' and 'support' and
        `run_metadata` keys.
    """
    df_metrics = pd.DataFrame(
        data=precision_recall_fscore_support(y_test, y_pred, zero_division=0),
        index=['precision', 'recall', 'fscore', 'support'],
        columns=target_names,
    ).T
    df_metrics.loc['total_weighted'] = precision_recall_fscore_support(
        y_true=y_test,
        y_pred=y_pred,
        average='weighted',
        zero_division=0,
    )
    df_metrics.loc['total_weighted', 'support'] = df_metrics['support'].sum()
    df_metrics['support'] = df_metrics['support'].astype(int)
    df_metrics = df_metrics.reset_index().rename({'index': 'target'}, axis=1)
    for k, v in metadata.items():
        df_metrics[k] = v
    return df_metrics


def aggregate_run_metrics(df_list: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate metrics across runs within the same experiment to determine averages and standard deviations.

    Parameters
    ----------
    df_list : list[pd.DataFrame]
        A list of dataframes containing run metrics.

    Returns
    -------
    df_metrics : pd.DataFrame
        Aggregated metrics dataframe with averaged performance and standard deviations within each experiment.
    """
    df_runs = pd.concat(df_list, axis=0, ignore_index=True)
    df_runs.drop('run_number', axis=1, inplace=True)
    to_groupby = ['experiment_id', 'train_size', 'parameters', 'target']

    # average and compute standard deviations
    df_metrics_avg = df_runs.groupby(to_groupby).mean()
    df_metrics_std = df_runs.groupby(to_groupby)[['precision', 'recall', 'fscore']].std().fillna(0.)
    df_metrics = df_metrics_avg.join(df_metrics_std, rsuffix='_std').reset_index()
    return df_metrics
