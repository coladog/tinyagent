import torch.nn as nn


maximum_seq_length = 20
minimum_seq_length = 1

dataset_size = 10000

def get_model():
    gru = nn.GRU(1, 4, 1, batch_first=True)
    mlp = nn.Sequential(
        nn.Linear(4, 4),
        nn.Sigmoid(),
        nn.Linear(4, 2)
    )
    return gru, mlp
