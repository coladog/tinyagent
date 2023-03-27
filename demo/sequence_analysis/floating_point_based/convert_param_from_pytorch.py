import torch
import struct

from util import *


def get_param_from_pytorch(model):
    keys = model.state_dict().keys()
    parameters = []
    for key in keys:
        p = model.state_dict()[key].flatten().tolist()
        parameters.extend(p)
    return parameters

def write_param_for_tinyagent(param, file):
    with open(file, "wb") as f:
        for i in param:
            f.write(struct.pack("f", i))


if __name__ == "__main__":
    gru, mlp = get_model()
    gru.load_state_dict(torch.load(GRU_PARAM_PATH))
    mlp.load_state_dict(torch.load(MLP_PARAM_PATH))

    write_param_for_tinyagent(get_param_from_pytorch(gru), GRU_PARAM_TA_PATH)
    write_param_for_tinyagent(get_param_from_pytorch(mlp), MLP_PARAM_TA_PATH)

    print("[+] converted param from PyTorch to TinyAgent:")
    print("[+] \t", GRU_PARAM_PATH, "->", GRU_PARAM_TA_PATH)
    print("[+] \t", MLP_PARAM_PATH, "->", MLP_PARAM_TA_PATH)
