def consturct_dataset(bunch, target2label: dict) -> list[dict]:
    examples = list()
    label_names = sorted(set(target2label.values()))
    for text, target in zip(bunch['data'], bunch['target']):
        target_name = bunch['target_names'][target]
        label_name = target2label[target_name]
        example = {'text': text, 'label': label_names.index(label_name)}
        examples.append(example)
    return examples
