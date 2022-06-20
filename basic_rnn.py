#%%
from ast import arg
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from utils import check_cuda, read_process_text, one_hot_encode, sample, train_model
from models import *
import argparse as ap
#%%
parser = ap.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.01)
parser.parse_args()
args = parser.parse_args()
input_seq, target_seq, char2int, int2char, maxlen, text = read_process_text(
    "tiny-shakespeare.txt", subset=4
)
dict_size = len(char2int)
seq_len = maxlen - 1
batch_size = len(text)
#%%
input_seq = one_hot_encode(input_seq, dict_size, seq_len, batch_size)
print(
    "Input shape: {} --> (Batch Size, Sequence Length, One-Hot Encoding Size)".format(
        input_seq.shape
    )
)
input_seq = torch.from_numpy(input_seq)
target_seq = torch.Tensor(target_seq)
#%%
device = check_cuda()
#%%
# Instantiate the model with hyperparameters
model = Model(input_size=dict_size, output_size=dict_size, hidden_dim=12, n_layers=1, device=device)
model = model.to(device)
# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#%%
# Training Run
input_seq = input_seq.to(device)
trained_model = train_model(model, input_seq, target_seq, criterion, optimizer, device, epochs=args.epochs)
#%%
sampleOut = sample(model, 30, "First", char2int, int2char, dict_size, device)
print(sampleOut)
# %%