"""
This module defines miscellaneous utility functions for preparing the data.
"""
# standard library
import re
from collections import Counter

# data wrangling
from datasets import DatasetDict


def split_camel_case(string: str) -> str:
    """
    Split a camel case string into separate words.
    Parameters
    ----------
    string : str
        Input string to be split.

    Returns
    -------
    str
        Input string split with space-separated words.

    Examples
    --------
    >>> split_camel_case('MeanOfTransportation')
    'Mean Of Transportation'
    >>> split_camel_case('Village')
    'Village'
    """
    assert ' ' not in string, f'Expected a single string in CamelCase. Got "{string}"'
    pattern = r'([A-Z][a-z]+)'
    tokens = [token for token in re.split(pattern, string) if token]
    string = ' '.join(tokens).strip()
    return string


def describe_labels(dataset_dict: DatasetDict):
    """
    Describe the distribution of class labels in train and test splits by printing a table.

    Parameters
    ----------
    dataset_dict : DatasetDict
        Dataset dict from Hugging Face.

    Returns
    -------
    None
    """
    counters = dict()
    for split_name in ('train', 'test'):
        if 'label' in dataset_dict[split_name].features:
            label_feature = dataset_dict[split_name].features['label']
            labels = map(label_feature.int2str, dataset_dict[split_name]['label'])
            counters[split_name] = Counter(labels)
        elif 'labels' in dataset_dict[split_name].features:
            label_feature = dataset_dict[split_name].features['labels'].feature
            labels_list = [map(label_feature.int2str, labels) for labels in dataset_dict[split_name]['labels']]
            counters[split_name] = Counter([label for labels in labels_list for label in labels])
        else:
            raise ValueError(f'Expected label or labels in {split_name} split.')

    max_len = max(map(len, label_feature.names))
    print('Class count: {:,}'.format(len(counters['train'])))
    print('{:<{max_len}}\t{}\t{}\t{}'.format('Label', 'Train', 'Test', 'Support (Train)', max_len=max_len))
    for label, support in counters['train'].most_common():
        share_train = counters['train'][label] / sum(counters['train'].values()) * 100
        share_test = counters['test'][label] / sum(counters['test'].values()) * 100
        print('{:<{max_len}}\t{:.2f}%\t{:.2f}%\t{:,}'.format(label, share_train, share_test, support, max_len=max_len))
