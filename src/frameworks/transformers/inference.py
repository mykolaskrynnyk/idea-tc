
def predict(model, dataset) -> list[int]:
    outputs = model.predict(dataset)
    y_pred=outputs.predictions.argmax(axis=-1)
    return y_pred
