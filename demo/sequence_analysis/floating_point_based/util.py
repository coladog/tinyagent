import torch.nn as nn


GRU_PARAM_PATH = "../pytorch_model_generator/data/gru.model"
MLP_PARAM_PATH = "../pytorch_model_generator/data/mlp.model"

GRU_PARAM_TA_PATH = "./data/gru.ta"
MLP_PARAM_TA_PATH = "./data/mlp.ta"

# don't generate sequence length > MAX_SEQ_LENGTH
MAX_SEQ_LENGTH = 20
data_to_test = [
    [1, 0, 1, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
]

def get_model():
    gru = nn.GRU(1, 4, 1, batch_first=True)
    mlp = nn.Sequential(
        nn.Linear(4, 4),
        nn.Sigmoid(),
        nn.Linear(4, 2)
    )
    return gru, mlp