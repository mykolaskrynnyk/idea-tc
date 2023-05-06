import torch


def get_device() -> str:
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'  # Apple Silicon
    else:
        device = 'cpu'
    return device


def collate_batch_variable(batch):
    device = get_device()
    label_list, text_list, offsets = [], [], [0]
    for example in batch:
        label_list.append(example['label'])
        text_list.append(example['tokens'])
        offsets.append(example['tokens'].size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


def collate_batch_fixed(batch):
    device = get_device()
    label_list, text_list = [], []
    for example in batch:
        label_list.append(example['label'])
        text_list.append(example['tokens'])
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    return label_list.to(device), text_list.to(device)


def count_parameters(model) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.

    Parameters
    ----------
    model
        PyTorch model.

    Returns
    -------
    count : int
        Number of trainable parameters in the model.
    """
    count = sum(p.numel() for p in model.parameters())
    return count
