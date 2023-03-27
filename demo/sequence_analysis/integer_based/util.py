import torch.nn as nn


FLOAT_RANGE = 10 # all values belongs to [-10, 10]
SCALE_BIT = 30
SCALE = FLOAT_RANGE / (2 ** SCALE_BIT)
TABLE_RANGE = 10000

SIGMOID_TABLE_INPUT = "./data/sigmoid_table_input.ta"
SIGMOID_TABLE_OUTPUT = "./data/sigmoid_table_output.ta"

TANH_TABLE_INPUT = "./data/tanh_table_input.ta"
TANH_TABLE_OUTPUT = "./data/tanh_table_output.ta"

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
