
# standard library
import os
import re
from xml.etree import ElementTree
from collections import defaultdict

# data wrangling
import pandas as pd


def _get_books(file_path: str) -> list[dict]:
    books = list()
    with open(file_path, 'r') as file:
        book = list()
        for line in file:
            line = re.sub(r'&', 'and', line)  # cannot have & inside xml
            book.append(line)
            if line == '</book>\n':
                books.append(''.join(book))
                book.clear()
    return books


def _parse_metadata(metadata) -> dict:
    metadata_dict = dict()
    for element in metadata:
        if element.tag != 'topics':
            metadata_dict[element.tag] = element.text
        else:
            metadata_dict[element.tag] = [topic.text for topic in element]
    return metadata_dict


def _parse_book(book: str) -> dict:
    book_dict = dict()
    tree = ElementTree.fromstring(book)
    for element in tree:
        if element.tag != 'metadata':
            book_dict[element.tag] = element.text
        else:
            matadata = _parse_metadata(element)
            book_dict.update(matadata)
    return book_dict


def read_books(file_path: str) -> pd.DataFrame:
    books = _get_books(file_path)
    df_books = pd.DataFrame(map(_parse_book, books))
    return df_books


def read_hierarchy(file_path: str) -> dict:
    hierarchy = defaultdict(list)
    with open(file_path, 'r') as file:
        for line in file:
            # use unpacking to handle cases where no subtopic exists, e.g., humor and poetry
            topic, *subtopic = line.replace(r'&', 'and').strip().split('\t')
            hierarchy[topic].extend(subtopic)
    return hierarchy


def read_dataset(folder_path: str) -> pd.DataFrame:
    file_path = os.path.join(folder_path, 'BlurbGenreCollection_EN_train.txt')
    df_books = read_books(file_path)

    file_path = os.path.join(folder_path, 'hierarchy.txt')
    hierarchy = read_hierarchy(file_path)

    # level 0 topics, these are not subtopics of any topic
    topics_0 = set(hierarchy) - {subtopic for subtopics in hierarchy.values() for subtopic in subtopics}
    assert len(topics_0) == 7, f'Expected 7 level 0 topics, obtained {len(topics_0)}.'

    # level 1 topics, these are direct subtopics of level 0 topics, use humor and poetry as topic 1 for themselves
    topics_1 = {topic_1 for topic_0 in topics_0 for topic_1 in hierarchy.get(topic_0, [topic_0])}
    assert len(topics_1) == 52, f'Expected 52 level 1 topics, obtained {len(topics_1)}.'

    df_books['topics_0'] = df_books['topics'].apply(lambda topics: set(topics) & topics_0).apply(list)
    df_books['topics_1'] = df_books['topics'].apply(lambda topics: set(topics) & topics_1).apply(list)
    df_books = df_books.reindex(['body', 'topics_1'], axis=1)
    df_books.rename({'body': 'text', 'topics_1': 'labels'}, axis=1, inplace=True)
    df_books = df_books.loc[df_books['labels'].str.len().ge(1)].copy()
    assert df_books.isna().sum().sum() == 0, 'Unexpected missing values'
    return df_books
