#%%
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from utils import check_cuda, read_process_text, one_hot_encode, sample, textDatasetFromFile, train_model
from torch.utils.data import Dataset, DataLoader
from archs import *
import argparse as ap
#%%
parser = ap.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--predict", action="store_true")
parser.parse_args()
args = parser.parse_args()
input_seq, target_seq, char2int, int2char, maxlen, text = read_process_text(
    "tiny-shakespeare.txt", subset=10000
)
dict_size = len(char2int)
seq_len = maxlen - 1
# batch_size = len(text)
batch_size = 300
#%%
input_seq = one_hot_encode(input_seq, dict_size, seq_len, batch_size)
print(
    "Input shape: {} --> (Batch Size, Sequence Length, One-Hot Encoding Size)".format(
        input_seq.shape
    )
)
target_seq = torch.Tensor(target_seq)

traindataset = textDatasetFromFile(input_seq, target_seq)
dataloader = DataLoader(traindataset, batch_size=batch_size,
                        shuffle=True, num_workers=10)

#%%
device = check_cuda()
#%%
# Instantiate the model with hyperparameters
model = SimpleRNN(input_size=dict_size, hidden_size=12, num_layers=2, bias=True, output_size=dict_size, activation="tanh")
# model = LSTM(input_size=dict_size, hidden_size=512, num_layers=2, bias=True, output_size=dict_size)
# model = GRU(input_size=dict_size, hidden_size=12, num_layers=2, bias=True, output_size=dict_size)
# model = BidirRecurrentModel(mode = "LSTM",input_size=dict_size, hidden_size=12, num_layers=2, bias=True, output_size=dict_size)
# model = GRU(input_size=dict_size, hidden_size=12, num_layers=2, bias=True, output_size=dict_size)
model = model.to(device)
# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#%%
# Training Run
if not args.predict:
    trained_model = train_model(model, dataloader, criterion, optimizer, device, epochs=args.epochs)
else:
    trained_model = torch.load("model.h5")
#%%
sampleOut = sample(model, 100, "First", char2int, int2char, dict_size, device)
print(sampleOut)
# %%