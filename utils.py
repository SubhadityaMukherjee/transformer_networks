import torch
from torch import nn
import numpy as np
from tqdm import tqdm


def read_process_text(filename="tiny-shakespeare.txt", subset=None):
    text = open("tiny-shakespeare.txt", "r").readlines()
    if subset!= None:
        text = text[:subset]
    # Join all the sentences together and extract the unique characters from the combined sentences
    chars = set("".join(text))
    # Creating a dictionary that maps integers to the characters
    int2char = dict(enumerate(chars))
    # Creating another dictionary that maps characters to integers
    char2int = {char: ind for ind, char in int2char.items()}
    maxlen = len(max(text, key=len))
    print("The longest string has {} characters".format(maxlen))
    # Padding

    # A simple loop that loops through the list of sentences and adds a ' ' whitespace until the length of the sentence matches
    # the length of the longest sentence
    for i in range(len(text)):
        while len(text[i]) < maxlen:
            text[i] += " "
    # Creating lists that will hold our input and target sequences
    input_seq = []
    target_seq = []

    for i in range(len(text)):
        # Remove last character for input sequence
        input_seq.append(text[i][:-1])

        # Remove firsts character for target sequence
        target_seq.append(text[i][1:])

    for i in range(len(text)):
        input_seq[i] = [char2int[character] for character in input_seq[i]]
        target_seq[i] = [char2int[character] for character in target_seq[i]]
    return input_seq, target_seq, char2int, int2char, maxlen, text


def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)

    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1
    return features


def check_cuda():
    is_cuda = torch.cuda.is_available()
    #%%
    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    return device

def predict(model, character, char2int, int2char, dict_size, device):
    # One-hot encoding our input to fit into the model
    character = np.array([[char2int[c] for c in character]])
    character = one_hot_encode(character, dict_size, character.shape[1], 1)
    character = torch.from_numpy(character)
    character = character.to(device)

    out = model(character)
    # print("sh", out.shape)
    prob = [nn.functional.softmax(out[j], dim=0).data for j in range(out.shape[0])]
    char_ind = [int2char[torch.max(j, dim=0)[1].item()] for j in prob]
    return "".join(char_ind)

    # prob = nn.functional.softmax(out[-1], dim=0).data

    # print("sh",prob.shape)
    # Taking the class with the highest probability score from the output
    # char_ind = torch.max(prob, dim=0)[1].item()

    # return int2char[char_ind]


#%%
def sample(model, out_len, start="hey", char2int=None, int2char=None, dict_size=None, device=None):
    model.eval()  # eval mode
    start = start.lower()
    # First off, run through the starting characters
    # chars = [ch for ch in start]
    size = out_len
    preds = ""

    char = predict(model, list(start), char2int, int2char, dict_size, device)
    # Now pass in the previous characters and get a new one
    for ii in range(size):
        char = predict(model, list(char), char2int, int2char, dict_size, device)
        # chars.extend(char)
        preds += char
    return f"{start} {preds[:out_len]}"

def train_model(model, input_seq, target_seq, criterion, optimizer, device, epochs=100):
    pbar = tqdm(range(1, epochs + 1), total=epochs)
    for epoch in pbar:
        optimizer.zero_grad()  # Clears existing gradients from previous epoch
        # input_seq = input_seq.to(device)
        output = model(input_seq)
        output = output.to(device)
        target_seq = target_seq.to(device)
        loss = criterion(output, target_seq.view(-1).long())
        loss.backward()  # Does backpropagation and calculates gradients
        optimizer.step()  # Updates the weights accordingly
        pbar.set_postfix({"loss": loss.item()})
    return model