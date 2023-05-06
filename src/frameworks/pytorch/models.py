# deep learning
import torch
from torch import nn
import torch.nn.functional as F

# local packages
from .utils import get_device


class DeepAveragingNetwork(nn.Module):
    """
    The implementation of DAN (Iyyer et al., 2015) in PyTorch. See https://aclanthology.org/P15-1162.
    """
    def __init__(
            self,
            num_class: int,
            vocab_size: int,
            embed_dim: int = 300,
            dropout: float = .3,
            hidden_dims: list[int] = (300, 300),
    ):
        super(DeepAveragingNetwork, self).__init__()
        self.embedding = nn.EmbeddingBag(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            sparse=False,
            mode='mean',
        )
        self.dropout = nn.Dropout(p=dropout)
        modules = []
        for idx, hidden_dim in enumerate(hidden_dims):
            in_features = embed_dim if idx == 0 else hidden_dims[idx-1]
            modules.append(nn.Linear(in_features=in_features, out_features=hidden_dim, bias=True))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(in_features=hidden_dims[-1], out_features=num_class))
        self.fcs = nn.Sequential(*modules)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        embedded = self.dropout(embedded)
        output = self.fcs(embedded)
        return output


class RecurrentNeuralNetwork(nn.Module):
    """
    Simple LSTM for text classification.
    """
    def __init__(self, num_class: int, vocab_size: int, embed_dim: int = 300, hidden_dim: int = 128):
        super(RecurrentNeuralNetwork, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            sparse=False,
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,

        )
        self.fc = nn.Linear(
            in_features=hidden_dim,
            out_features=num_class,
            bias=True,
        )

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        output = self.fc(output[:, -1, :])
        return output


class ConvolutionalNeuralNetwork(nn.Module):
    """
    The implementation of a CNN for text classification based on the recommendations from (Zhang and Wallace, 2017).
    See https://aclanthology.org/I17-1026.
    """
    def __init__(
            self,
            num_class: int,
            vocab_size: int,
            embed_dim: int = 300,
            out_channels: int = 100,
            kernel_sizes: tuple[int, ...] = (3, 4, 5),
            dropout: float = 0.5
    ):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            sparse=False,
        )
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=out_channels,
                kernel_size=kernel_size
            ) for kernel_size in kernel_sizes
        ])
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(
            in_features=out_channels * len(kernel_sizes),
            out_features=num_class,
            bias=True,
        )

    def forward(self, text):
        # batch_size, seq_len, embed_dim
        embedded = self.embedding(text)

        # batch_size, embed_dim, seq_len
        embedded = embedded.permute(0, 2, 1)

        # len(self.convs), batch_size, out_channels, seq_len
        features_list = [F.relu(conv1d(embedded)) for conv1d in self.convs]

        # len(self.convs), batch_size, out_channels, 1
        pooled_list = [F.max_pool1d(feature, kernel_size=feature.shape[-1]) for feature in features_list]

        # batch_size, out_channels * len(self.convs)
        hidden = torch.cat([pooled.squeeze(dim=2) for pooled in pooled_list], dim=1)

        # batch_size, num_class
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        return output


def get_cnn(num_class: int, vocab_size: int):
    device = get_device()
    model = ConvolutionalNeuralNetwork(
        num_class=num_class,
        vocab_size=vocab_size,
        embed_dim=300,
        out_channels=50,
        kernel_sizes=(3, 4, 5),
    ).to(device)
    return model
