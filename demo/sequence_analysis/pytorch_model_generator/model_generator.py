import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import pickle

from torch.utils.data import DataLoader
from data_generator import *
from util import *

def train():
    with open("data/train.pkl", "rb") as f:
        data = pickle.load(f)

    data_loader = DataLoader(
        TransitionData(data),
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn,
    )

    gru, mlp = get_model()

    optimizers = [
        torch.optim.Adam(gru.parameters(), lr=1e-3),
        torch.optim.Adam(mlp.parameters(), lr=1e-3),
    ]

    criterion = nn.MSELoss()

    for epoch in range(30):
        total_loss, batch_cnt = 0, 0
        for _, (batch_x, label, batch_length) in enumerate(data_loader):
            batch_x_pack = rnn_utils.pack_padded_sequence(
                batch_x, batch_length, batch_first=True
            )

            h0 = torch.zeros((1, len(batch_x), 4))
            _, h0 = gru(batch_x_pack, h0)
            predict = mlp(h0.squeeze())

            loss = criterion(predict, label)
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            total_loss += loss.item()
            batch_cnt += 1
        print("[+] Training {:>2}/30: loss: {:.5f}".format(epoch+1, total_loss/batch_cnt))
    
    torch.save(gru.state_dict(), "data/gru.model")
    torch.save(mlp.state_dict(), "data/mlp.model")
    print("[+] Saved trained models into data/gru.model & data/mlp.model")


if __name__ == "__main__":
    train()
