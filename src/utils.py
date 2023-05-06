"""
Miscellaneous routines for showing the data.
"""
# standard library
from pathlib import Path
from typing import Literal

# data wrangling
import pandas as pd
from datasets import load_from_disk


# generic type hints
feature_types = Literal['text', 'text_clean', 'input_ids']
dataset_types = Literal['data.train', 'data.valid']
metric_types = Literal['fscore', 'precision', 'recall']


def show_datasets(path_datasets: Path) -> pd.DataFrame:
    """
    Describe processed datasets in terms of number of examples and classes.

    Parameters
    ----------
    path_datasets : Path
        Pathlib Path object to the directory where processed datasets are stored.

    Returns
    -------
    df_datasets : pd.DataFrame
        Pandas DataFrame describing each dataset's split sizes and number of classes.
    """
    def _get_stats(path: Path):
        dataset = load_from_disk(str(path))
        num_rows = dataset.num_rows
        num_classes = len(dataset['train'].features['label'].names)
        record = {
            'dataset': path.name,
            'num_rows_train': num_rows['train'],
            'num_rows_valid': num_rows.get('validation', 0),
            'num_rows_test': num_rows['test'],
            'num_classes': num_classes,
        }
        return record
    df_datasets = pd.DataFrame([_get_stats(path) for path in path_datasets.glob('*processed')])
    df_datasets.sort_values('num_rows_train', ascending=False, ignore_index=True, inplace=True)
    return df_datasets
