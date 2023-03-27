import torch
from util import *


if __name__ == "__main__":
    gru, mlp = get_model()
    gru.load_state_dict(torch.load(GRU_PARAM_PATH))
    mlp.load_state_dict(torch.load(MLP_PARAM_PATH))

    print("[+] read param from:")
    print("[+] \t ", GRU_PARAM_PATH)
    print("[+] \t ", MLP_PARAM_PATH)
    print("[+]")

    input_size, hidden_size = 3, 4

    for data in data_to_test:
        print("[+] now testing sequence: ", data)
        with torch.no_grad():
            data_t = torch.tensor(data).float().unsqueeze(1)
            h0 = torch.zeros((1, 1, 4)).squeeze(0)
            _, h0 = gru(data_t, h0)
            prediction = mlp(h0.squeeze()).tolist()
            print("[+] \t prediction: [{:.3f}, {:.3f}] -> number of 1: {:.0f} | length of sequence: {:.0f}".format(
                prediction[0], prediction[1], prediction[0] * len(data), prediction[1] * MAX_SEQ_LENGTH))

