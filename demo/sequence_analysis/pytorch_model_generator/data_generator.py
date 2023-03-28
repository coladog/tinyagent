import torch
import torch.nn.utils.rnn as rnn_utils
import torch.utils.data as data
import pickle
import random

from util import *

def generate_one_sequence():
    length = random.randint(minimum_seq_length, maximum_seq_length)
    one_generate_rate = random.randint(1, 10)
    seq = [
        [1 if random.randint(1, 10) <= one_generate_rate else 0]
        for _ in range(length)
    ]
    label = (sum([i[0] for i in seq]) / length, length / maximum_seq_length)
    return seq, label

def generate_dataset():
    train_set_size = int(dataset_size)
    train_set_save_path = "data/train.pkl"
    
    for set_size, save_path in [
        (train_set_size, train_set_save_path),
    ]:
        data = []
        for _ in range(set_size):
            data.append(generate_one_sequence())
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

class TransitionData(data.Dataset):
    def __init__(self, data_seq):
        self.data_seq = data_seq

    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        return self.data_seq[idx][0], self.data_seq[idx][1]

def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    batch_x, true_prediction = (
        [i[0] for i in data],
        [list(i[1]) for i in data],
    )
    batch_length = torch.tensor([len(i) for i in batch_x])
    batch_x = [torch.tensor(i).float() for i in batch_x]
    true_prediction = torch.tensor(true_prediction).float()
    batch_x = rnn_utils.pad_sequence(batch_x, batch_first=True, padding_value=0)

    return batch_x, true_prediction, batch_length


if __name__ == "__main__":
    generate_dataset()
    print("[+] generated training dataset into pytorch_model_generator/data/train.pkl")