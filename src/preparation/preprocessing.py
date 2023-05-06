"""
This module defines miscellaneous utility functions for preparing the data.
"""
# standard library
from pprint import pprint
from typing import Optional

# data wrangling
from datasets import DatasetDict, features

# nlp
from spacy.lang.en import English
from gensim.models.phrases import Phrases
from gensim.utils import simple_preprocess

# utils
from tqdm import tqdm

# local package
from .utils import describe_labels


def recode_class_labels(dataset_dict: DatasetDict, class_names: list[str]) -> DatasetDict:
    """
    Recode class names in all datasets within a dataset dictionary.

    Note that the input is mutated.

    Parameters
    ----------
    dataset_dict : DatasetDict
        Dataset dict from Hugging Face.
    class_names
        List of new class names to assign.

    Returns
    -------
    dataset_dict : DatasetDict
        Input dataset dict with recoded class names
    """
    class_label = features.ClassLabel(num_classes=len(class_names), names=class_names)
    for split_name in dataset_dict:
        dataset_dict[split_name].features['label'] = class_label
    return dataset_dict


def filter_and_lemmatise_texts(batch: list[dict], nlp: English) -> list[dict]:
    allowed_pos = {'NOUN', 'VERB', 'ADJ', 'CCONJ', 'SCONJ'}
    texts_clean = list()
    for doc in nlp.pipe(batch['text']):
        text_clean = ' '.join([token.lemma_ for token in doc if token.pos_ in allowed_pos])
        texts_clean.append(text_clean)
    batch['text_clean'] = texts_clean
    return batch


def tokenise_texts(batch: list[dict]) -> list[dict]:
    batch['text_clean'] = list(map(lambda text: simple_preprocess(text, deacc=True), batch['text_clean']))
    return batch


def get_model_phrases() -> Phrases:
    phrases = Phrases(
        min_count=2,
        threshold=.3,  # based on collocation score between -1 to 1
        max_vocab_size=100_000,
        scoring='npmi',
    )
    return phrases


def constuct_bigrams(batch: list[dict], phrases: Phrases) -> list[dict]:
    batch['text_clean'] = [' '.join(tokens) for tokens in phrases[batch['text_clean']]]
    return batch


def undersample_test_split(dataset_dict: DatasetDict, samples_per_class: int = 500, seed: int = 42) -> DatasetDict:
    test_size = dataset_dict['test'].features['label'].num_classes * samples_per_class
    dataset_dict['test'] = dataset_dict['test'].train_test_split(
        test_size=test_size,
        stratify_by_column='label',
        seed=seed,
    )['test']
    return dataset_dict


def preprocess_dataset(
        dataset_dict: DatasetDict,
        class_names: list[str],
        undersample_test: bool,
        nlp: English,
        phrases_save_path: Optional[str] = None,
        verbose: bool = True
) -> DatasetDict:
    if class_names is not None:
        dataset_dict = recode_class_labels(dataset_dict=dataset_dict, class_names=class_names)
    if verbose:
        pprint(dataset_dict['train'].features)
    if undersample_test:
        dataset_dict = undersample_test_split(dataset_dict=dataset_dict, samples_per_class=500)

    dataset_dict = dataset_dict.map(
        function=filter_and_lemmatise_texts,
        batched=True,
        batch_size=64,
        fn_kwargs={'nlp': nlp},
    )

    dataset_dict = dataset_dict.map(
        function=tokenise_texts,
        batched=True,
        batch_size=64,
        num_proc=4,
    )

    phrases = get_model_phrases()
    for batch in tqdm(dataset_dict['train'].iter(batch_size=32), disable=not verbose):
        phrases.add_vocab(batch['text_clean'])
    if verbose:
        print(f'Phrase count: {len(phrases.export_phrases()):,}')

    phrases = phrases.freeze()
    if phrases_save_path is not None:
        phrases.save(str(phrases_save_path))

    dataset_dict = dataset_dict.map(
        function=constuct_bigrams,
        batched=True,
        batch_size=64,
        fn_kwargs={'phrases': phrases},
    )

    if verbose:
        describe_labels(dataset_dict)
    return dataset_dict
