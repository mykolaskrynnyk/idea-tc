import pandas as pd


def get_labels_from_mesh(terms: str) -> list[str]:
    """
    Examples
    --------
    >>> get_labels_from_mesh('Animal; Glucose/*; Milk Proteins;')
    ['Animal', 'Glucose', 'Milk Proteins']
    """
    labels = set()
    for term in terms.split(';'):
        if not term.strip():
            continue
        # keep only a top-level term
        label = term.split('/')[0].split(', ')[0].strip(' .,')
        labels.add(label)
    return sorted(labels)


def clean_ohsumed_split(df_split: pd.DataFrame) -> pd.DataFrame:
    df_split['text'] = df_split.apply(lambda row: row['title'] + '\n' + row['abstract'], axis=1)
    df_split['labels'] = df_split['mesh_terms'].apply(get_labels_from_mesh)
    df_split = df_split.reindex(['text', 'labels'], axis=1)
    return df_split
