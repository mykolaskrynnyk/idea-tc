# deep learning
import torch

# utils
from tqdm import tqdm


def predict(model, dataloader, progress_bar: bool = False) -> list[int]:
    y_pred = list()
    for label, *inputs in tqdm(dataloader, disable=not progress_bar):
        with torch.no_grad():
            predicted_label = model(*inputs)
            y_batch = predicted_label.argmax(dim=1).cpu().detach().numpy().tolist()
            y_pred.extend(y_batch)
    return y_pred
