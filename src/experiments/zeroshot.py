"""
Routines allowing to experiment with scikit-learn and fastText models for supervised training and text classification.
"""
# standard library
import json
from typing import Any

# data wrangling
import xarray as xr
import pandas as pd
from datasets import DatasetDict
from scipy.special import softmax

# deep learning
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# utils
from tqdm import tqdm

# local packages
from . import evaluation, utils


def run_experiment(
        dataset_dict: DatasetDict,
        candidate_labels: list[str],
        checkpoint: str,
        device: Any,
        hypothesis_template: str = 'This example is {}.',
        progress_bar: bool = True,
        experiment_id: str = 'zero_shot',
        save_path: str = '',
) -> bool:
    """
    Run a series of experiments using "classic" approach, i.e., models from scikit-learn and fastText.

    Parameters
    ----------
    dataset_dict : DatasetDict
        Dataset dictionary object with a 'train', 'test' and (optionally) 'validation' splits.
    candidate_labels : list[str]
        List of candidate labels for zero-shot classification.
    checkpoint : str
        Checkpoint name from Transformers hub that is suitable for zero-shot text classification.
    device : Any
        Device name, such as 'cpu' or 'gpu' or `torch.device` to place the model on.
    hypothesis_template : str,
        The template used to turn each label into an NLI-style hypothesis. The default corresponds to the default
        value user in Transformers.
    progress_bar : bool, default=True
        If True, use a tqdm progress bar over the sample sizes.
    experiment_id : str, default='zero_shot'
        A string ID to be assigned to all metrics of the experiment.
    save_path : str, default=''
        If not None, it must be a path including a filename prefix (but no extension) to save experiments and
        predictions to.

    Returns
    -------
    True after the experiment results have been written to disk.
    """
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    tokeniser = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)  # use the original implementation, not Rust
    parameters = {
        'checkpoint': checkpoint,
        'multilabel': False,
        'number_parameterts': sum(p.numel() for p in model.parameters()),
    }
    metadata = {
        'experiment_id': experiment_id,
        'train_size': 0,
        'parameters': json.dumps(parameters, default=str),
        'run_number': 0,
    }
    logits_list = list()
    model.eval()
    for text in tqdm(dataset_dict['test']['text'], disable=not progress_bar):
        with torch.no_grad():
            batch = utils.prepare_zsc_batch(
                tokeniser=tokeniser,
                text=text,
                candidate_labels=candidate_labels,
                hypothesis_template=hypothesis_template,
                device=device,
            )
            # n candidate_labels by 3 ('contradiction', 'entailment', 'scores')
            logits = model(**batch).logits.cpu().numpy()
            logits_list.append(logits)

    logits = xr.DataArray(
        data=logits_list,
        dims=('id', 'candidate_labels', 'classes'),
        coords={
            'candidate_labels': candidate_labels,
            'classes': ['contradiction', 'entailment', 'scores'],
        },
    )
    y_pred = softmax(logits.sel(classes='entailment'), axis=1).argmax(axis=1)
    df_predictions = pd.DataFrame({'y_true': dataset_dict['test']['label'], experiment_id: y_pred})
    df_metrics = evaluation.calculate_run_metrics(dataset_dict['test']['label'], y_pred, candidate_labels, metadata)
    df_metrics = evaluation.aggregate_run_metrics(df_list=[df_metrics])

    # save metrics and predicted labels to disk, also save the raw logit values to experiment with top-k accuracy
    df_metrics.to_csv(f'{save_path}_{experiment_id}_metrics.csv', index=False)
    df_predictions.to_csv(f'{save_path}_{experiment_id}_predictions.csv', index=False)
    logits.to_netcdf(f'{save_path}_{experiment_id}_logits.nc')

    return True
