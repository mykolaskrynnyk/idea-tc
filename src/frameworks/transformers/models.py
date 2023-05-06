from transformers import AutoModelForSequenceClassification


def get_transformer(model_name: str, num_class: int):
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_name,
        num_labels=num_class,
    )
    return model
