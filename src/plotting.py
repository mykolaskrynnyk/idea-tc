"""
Functions to plot performance metrics.
"""
# standard library
from math import ceil

# data wrangling
import pandas as pd

# visualisation
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# local packages
from .utils import metric_types

# other settings
pio.templates.default = 'plotly_white'


def get_colour_map() -> dict:
    colour_map = {
        'dummy_classifier': '#7F7F7F',
        'complement_naive_bayes': '#BCBD22',
        'sgd_classifier': '#2CA02C',
        'fasttext': '#D62728',
        'cnn': '#FF7F0E',
        'deberta': '#1F77B4',
    }
    return colour_map


def get_name_map() -> dict:
    name_map = {
        'dummy_classifier': 'Dummy Classifier',
        'complement_naive_bayes': 'Complement Naive Bayes',
        'sgd_classifier': 'SGD Classifier',
        'fasttext': 'fastText',
        'cnn': 'CNN',
        'deberta_v3_small_finetuned': 'DeBERTa v3 Small (Fine-tuned)',
        'deberta_v3_small_zeroshot': 'DeBERTa v3 Small (Zero-shot)',
        'deberta_v3_xsmall_zeroshot': 'DeBERTa v3 XSmall (Zero-shot)',
        'deberta_v3_base_zeroshot': 'DeBERTa v3 Base (Zero-shot)',
    }
    return name_map


def get_line_map() -> dict:
    line_map = {
        'deberta_v3_small_finetuned': 'solid',
        'deberta_v3_small_zeroshot': 'dash',
        'deberta_v3_xsmall_zeroshot': 'dot',
        'deberta_v3_base_zeroshot': 'longdashdot',
    }
    return line_map


def plot_performance_overall(df_metrics: pd.DataFrame, metric: metric_types = 'fscore', title: str = '') -> go.Figure:
    """
    Plot weighted metric across all classes for each model.

    Parameters
    ----------
    df_metrics : pd.DataFrame
         DataFrame of metrics as returned by `read_metrics`.
    metric : metric_types, default='fscore'
        Name of the metric to plot, i.e., 'fscore', 'precision', 'recall'.
    title : str, default=''
        Plot title.
    Returns
    -------
    fig : Figure
        Plotly figure object with a linechart per model.
    """
    data = list()
    colour_map = get_colour_map()
    name_map = get_name_map()
    line_map = get_line_map()

    mask = df_metrics['target'].eq('total_weighted') & ~df_metrics['experiment_id'].str.startswith('deberta')
    df_supervised = df_metrics.loc[mask].copy()
    y_range = (df_supervised['train_size'].min(), df_supervised['train_size'].max())
    experiment_ids = ['dummy_classifier', 'complement_naive_bayes', 'sgd_classifier', 'fasttext', 'cnn']
    for experiment_id in experiment_ids:
        mask = df_supervised['experiment_id'].eq(experiment_id)
        scatter = go.Scatter(
            x=df_supervised.loc[mask, 'train_size'],
            y=df_supervised.loc[mask, metric],
            name=name_map[experiment_id],
            mode='lines+markers',
            line={'color': colour_map[experiment_id]},
        )
        data.append(scatter)

    mask = df_metrics['target'].eq('total_weighted') & df_metrics['experiment_id'].str.startswith('deberta')
    df_zeroshot = df_metrics.loc[mask].copy()
    experiment_ids = ['deberta_v3_small_finetuned', 'deberta_v3_small_zeroshot', 'deberta_v3_xsmall_zeroshot', 'deberta_v3_base_zeroshot']
    for experiment_id in experiment_ids:
        mask = df_zeroshot['experiment_id'].eq(experiment_id)
        score = df_zeroshot.loc[mask, 'fscore'].item()
        scatter = go.Scatter(
            x=y_range,
            y=(score, score),
            mode='lines',
            name=name_map[experiment_id],
            line={'color': 'black', 'dash': line_map[experiment_id]},
            yaxis='y1'
        )
        data.append(scatter)

    layout = go.Layout(
        title=title,
        showlegend=True,
        xaxis={'title': '# Training Examples', 'type': 'log'},
        yaxis={'title': 'Weighted F1-Score', 'range': (0., 1.)},
    )

    fig = go.Figure(data=data, layout=layout)

    return fig


def plot_performance_overall_all(list_df_metrics: list[pd.DataFrame], subtitles: list[str], metric: metric_types = 'fscore', title: str = '', per_row : int = 5) -> go.Figure:
    """
    Plot weighted metric across all classes for each model.

    Parameters
    ----------
    df_metrics : pd.DataFrame
         DataFrame of metrics as returned by `read_metrics`.
    metric : metric_types, default='fscore'
        Name of the metric to plot, i.e., 'fscore', 'precision', 'recall'.
    title : str, default=''
        Plot title.
    Returns
    -------
    fig : Figure
        Plotly figure object with a linechart per model.
    """
    colour_map = get_colour_map()
    name_map = get_name_map()
    line_map = get_line_map()

    fig = make_subplots(
        rows=ceil(len(list_df_metrics) / per_row),
        cols=per_row,
        # shared_yaxes=True,
        subplot_titles=subtitles,
        vertical_spacing=0.15,
    )

    for idx, df_metrics in enumerate(list_df_metrics):
        i = idx // per_row + 1
        j = idx % per_row + 1
        mask = df_metrics['target'].eq('total_weighted') & ~df_metrics['experiment_id'].str.startswith('deberta')
        df_supervised = df_metrics.loc[mask].copy()
        y_range = (df_supervised['train_size'].min(), df_supervised['train_size'].max())
        experiment_ids = ['dummy_classifier', 'complement_naive_bayes', 'sgd_classifier', 'fasttext', 'cnn']
        for experiment_id in experiment_ids:
            mask = df_supervised['experiment_id'].eq(experiment_id)
            scatter = go.Scatter(
                x=df_supervised.loc[mask, 'train_size'],
                y=df_supervised.loc[mask, metric],
                name=name_map[experiment_id],
                mode='lines+markers',
                line={'color': colour_map[experiment_id]},
                showlegend=not idx,  # only the first iteration
            )
            fig.add_trace(scatter, row=i, col=j)

        mask = df_metrics['target'].eq('total_weighted') & df_metrics['experiment_id'].str.startswith('deberta')
        df_zeroshot = df_metrics.loc[mask].copy()
        experiment_ids = ['deberta_v3_small_finetuned', 'deberta_v3_small_zeroshot', 'deberta_v3_xsmall_zeroshot', 'deberta_v3_base_zeroshot']
        for experiment_id in experiment_ids:
            mask = df_zeroshot['experiment_id'].eq(experiment_id)
            score = df_zeroshot.loc[mask, 'fscore'].item()
            scatter = go.Scatter(
                x=y_range,
                y=(score, score),
                mode='lines',
                name=name_map[experiment_id],
                line={'color': 'black', 'dash': line_map[experiment_id]},
                yaxis='y1',
                showlegend=not idx,  # only the first iteration
            )
            fig.add_trace(scatter, row=i, col=j)
            last_row = len(list_df_metrics) - per_row
            fig.update_xaxes(title_text='# Training Examples' if idx >= last_row or last_row == 0 else '', type='log', row=i, col=j)
            fig.update_yaxes(title_text='Weighted F1-Score' if j == 1 else '', range=(0., 1.), row=i, col=j)
    return fig


def plot_performance_by_class(df_metrics: pd.DataFrame, metric: metric_types = 'fscore', title: str = '', per_row: int = 3, target_order: list[str] = None) -> go.Figure:
    """
    Plot weighted metric across all classes for each model.

    Parameters
    ----------
    df_metrics : pd.DataFrame
         DataFrame of metrics as returned by `read_metrics`.
    metric : metric_types, default='fscore'
        Name of the metric to plot, i.e., 'fscore', 'precision', 'recall'.
    title : str, default=''
        Plot title.
    per_row : int, default=3
        Number of plots per row.
    target_order : list[str]
        Order of subplots by target.
    Returns
    -------
    fig : Figure
        Plotly figure object with a linechart per model.
    """
    colour_map = get_colour_map()
    name_map = get_name_map()
    line_map = get_line_map()

    mask = df_metrics['target'].ne('total_weighted') & ~df_metrics['experiment_id'].str.startswith('deberta')
    df_supervised = df_metrics.loc[mask].copy()
    if target_order is None:
        targets = df_supervised['target'].unique().tolist()
    else:
        targets = target_order.copy()
    fig = make_subplots(rows=ceil(len(targets) / per_row), cols=per_row, shared_yaxes=True, subplot_titles=targets)
    y_range = (df_supervised['train_size'].min(), df_supervised['train_size'].max())
    experiment_ids = ['dummy_classifier', 'complement_naive_bayes', 'sgd_classifier', 'fasttext', 'cnn']
    for idx, target in enumerate(targets):
        i = idx // per_row + 1
        j = idx % per_row + 1
        for experiment_id in experiment_ids:
            mask = df_supervised['experiment_id'].eq(experiment_id) & df_supervised['target'].eq(target)
            scatter = go.Scatter(
                x=df_supervised.loc[mask, 'train_size'],
                y=df_supervised.loc[mask, metric],
                name=name_map[experiment_id],
                mode='lines+markers',
                line={'color': colour_map[experiment_id]},
                showlegend=not idx,  # only the first iteration
            )
            fig.add_trace(scatter, row=i, col=j)

    mask = df_metrics['target'].ne('total_weighted') & df_metrics['experiment_id'].str.startswith('deberta')
    df_zeroshot = df_metrics.loc[mask].copy()
    experiment_ids = ['deberta_v3_small_finetuned', 'deberta_v3_small_zeroshot', 'deberta_v3_xsmall_zeroshot', 'deberta_v3_base_zeroshot']
    for idx, target in enumerate(targets):
        i = idx // per_row + 1
        j = idx % per_row + 1
        for experiment_id in experiment_ids:
            mask = df_zeroshot['experiment_id'].eq(experiment_id) & df_zeroshot['target'].eq(target)
            score = df_zeroshot.loc[mask, 'fscore'].item()
            scatter = go.Scatter(
                x=y_range,
                y=(score, score),
                mode='lines',
                name=name_map[experiment_id],
                line={'color': 'black', 'dash': line_map[experiment_id]},
                showlegend=not idx,
                yaxis='y1',
                )
            fig.add_trace(scatter, row=i, col=j)
            last_row = len(targets) - per_row
            fig.update_xaxes(title_text='# Training Examples' if idx >= last_row or last_row == 0 else '', type='log', row=i, col=j)
            fig.update_yaxes(title_text='Weighted F1-Score' if j == 1 else '', range=(0., 1.), row=i, col=j)
    return fig


def plot_top_k_accuracy(df_accuracy: pd.DataFrame) -> go.Figure:
    fig = px.line(
        data_frame=df_accuracy,
        x='k',
        y='top_k_accuracy',
        color='experiment_id',
        facet_col='dataset',
        markers=True,
    )
    fig.update_layout(
        xaxis={'title': 'k'},
        yaxis={'title': 'Top k Accuracy', 'range': (0., 1.1)},
        legend={'title': 'Model'},
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1]))
    return fig


def plot_prompt_performance(df_promts: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        data_frame=df_promts,
        x='prompt',
        y='fscore',
        color='experiment_id',
        barmode='group',
    )
    fig.update_layout(
        xaxis={'title': 'Prompt Template'},
        yaxis={'title': 'Weighted F1-Score', 'range': (0., 1.)},
        legend={'title': 'Model'},
    )
    return fig
