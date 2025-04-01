# %% [markdown]
# # Privacy-Preserving Record Linkage (PPRL): Investigating Dataset Extension Attacks

# %% [markdown]
# ## Setup

# %%
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision

from utils import *

import matplotlib.pyplot as plt # For data viz
import pandas as pd
import hickle as hkl
import numpy as np
import string
import sys
from tqdm.notebook import tqdm

from graphMatching.gma import run_gma

from datasets.bloom_filter_dataset import BloomFilterDataset
from datasets.tab_min_hash_dataset import TabMinHashDataset
from datasets.two_step_hash_dataset_padding import TwoStepHashDatasetPadding
from datasets.two_step_hash_dataset_frequency_string import TwoStepHashDatasetFrequencyString
from datasets.two_step_hash_dataset_one_hot_encoding import TwoStepHashDatasetOneHotEncoding

from pytorch_models.bloom_filter_to_two_gram_classifier import BloomFilterToTwoGramClassifier
from pytorch_models.tab_min_hash_to_two_gram_classifier import TabMinHashToTwoGramClassifier
from pytorch_models.two_step_hash_to_two_gram_classifier import TwoStepHashToTwoGramClassifier

print('System Version:', sys.version)
print('PyTorch version', torch.__version__)
print('Torchvision version', torchvision.__version__)
print('Numpy version', np.__version__)
print('Pandas version', pd.__version__)

# %% [markdown]
# ## Run GMA

# %%
# Parameters
GLOBAL_CONFIG = {
    "Data": "./data/datasets/fakename_5k.tsv",
    "Overlap": 0.9,
    "DropFrom": "Both",
    "Verbose": True,  # Print Status Messages
    "MatchingMetric": "cosine",
    "Matching": "MinWeight",
    "Workers": -1,
    "SaveAliceEncs": False,
    "SaveEveEncs": False,
    "DevMode": True,
}

# For TSH: Padding / FrequencyString / OneHotEncoding
DEA_CONFIG = {
    "TSHMode": "OneHotEncoding",
    "DevMode": False,
}

ENC_CONFIG = {
    "AliceAlgo": "TwoStepHash",
    "AliceSecret": "SuperSecretSalt1337",
    "AliceN": 2,
    "AliceMetric": "dice",
    "EveAlgo": "None",
    "EveSecret": "ATotallyDifferentString42",
    "EveN": 2,
    "EveMetric": "dice",
    # For BF encoding
    "AliceBFLength": 1024,
    "AliceBits": 10,
    "AliceDiffuse": False,
    "AliceT": 10,
    "AliceEldLength": 1024,
    "EveBFLength": 1024,
    "EveBits": 10,
    "EveDiffuse": False,
    "EveT": 10,
    "EveEldLength": 1024,
    # For TMH encoding
    "AliceNHash": 1024,
    "AliceNHashBits": 64,
    "AliceNSubKeys": 8,
    "Alice1BitHash": True,
    "EveNHash": 1024,
    "EveNHashBits": 64,
    "EveNSubKeys": 8,
    "Eve1BitHash": True,
    # For 2SH encoding
    "AliceNHashFunc": 10,
    "AliceNHashCol": 1000,
    "AliceRandMode": "PNG",
    "EveNHashFunc": 10,
    "EveNHashCol": 1000,
    "EveRandMode": "PNG",
}

EMB_CONFIG = {
    "Algo": "Node2Vec",
    "AliceQuantile": 0.9,
    "AliceDiscretize": False,
    "AliceDim": 128,
    "AliceContext": 10,
    "AliceNegative": 1,
    "AliceNormalize": True,
    "EveQuantile": 0.9,
    "EveDiscretize": False,
    "EveDim": 128,
    "EveContext": 10,
    "EveNegative": 1,
    "EveNormalize": True,
    # For Node2Vec
    "AliceWalkLen": 100,
    "AliceNWalks": 20,
    "AliceP": 250,
    "AliceQ": 300,
    "AliceEpochs": 5,
    "AliceSeed": 42,
    "EveWalkLen": 100,
    "EveNWalks": 20,
    "EveP": 250,
    "EveQ": 300,
    "EveEpochs": 5,
    "EveSeed": 42
}

ALIGN_CONFIG = {
    "RegWS": max(0.1, GLOBAL_CONFIG["Overlap"]/2), #0005
    "RegInit":1, # For BF 0.25
    "Batchsize": 1, # 1 = 100%
    "LR": 200.0,
    "NIterWS": 100,
    "NIterInit": 5 ,  # 800
    "NEpochWS": 100,
    "LRDecay": 1,
    "Sqrt": True,
    "EarlyStopping": 10,
    "Selection": "None",
    "MaxLoad": None,
    "Wasserstein": True
}

# %%
eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash = get_hashes(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG)

if(os.path.isfile("./data/available_to_eve/reidentified_individuals_%s_%s_%s_%s.tsv" % (eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash)) & os.path.isfile("./data/available_to_eve/not_reidentified_individuals_%s_%s_%s_%s.h5" % (eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash))):
    #Load Disk From Data
    #TODO: Replace pd.read_csv with hkl.load
    reidentified_individuals = hkl.load('./data/available_to_eve/reidentified_individuals_%s_%s_%s_%s.h5' % (eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash))
    df_reidentified_individuals = pd.DataFrame(reidentified_individuals[1:], columns=reidentified_individuals[0])

    not_reidentified_individuals = hkl.load('./data/available_to_eve/not_reidentified_individuals_%s_%s_%s_%s.h5' % (eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash))
    df_not_reidentified_individuals = pd.DataFrame(not_reidentified_individuals[1:], columns=not_reidentified_individuals[0])

else:
    reidentified_individuals, not_reidentified_individuals = run_gma(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG, DEA_CONFIG, eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash)

    df_reidentified_individuals = pd.DataFrame(reidentified_individuals[1:], columns=reidentified_individuals[0])
    df_not_reidentified_individuals = pd.DataFrame(not_reidentified_individuals[1:], columns=not_reidentified_individuals[0])

# %% [markdown]
# ## Create Datasets

# %%
#Create the 2-grams with dictionary

#Generate all 2-grams
alphabet = string.ascii_lowercase

# Generate all letter-letter 2-grams (aa-zz)
alphabet = string.ascii_lowercase
letter_letter_grams = [a + b for a in alphabet for b in alphabet]

# Generate all digit-digit 2-grams (00-99)
digits = string.digits
digit_digit_grams = [d1 + d2 for d1 in digits for d2 in digits]

# Generate all letter-digit 2-grams (a0-z9)
letter_digit_grams = [l + d for l in alphabet for d in digits]

# Combine all sets
all_two_grams = letter_letter_grams  + letter_digit_grams + digit_digit_grams

# Get a dictionary associating each 2-gram with an index
two_gram_dict = {i: two_gram for i, two_gram in enumerate(all_two_grams)}

# %%
# Create Datasets based on chosen encoding
if ENC_CONFIG["AliceAlgo"] == "BloomFilter":
    data_labeled = BloomFilterDataset(df_reidentified_individuals, is_labeled=True, all_two_grams=all_two_grams, dev_mode=GLOBAL_CONFIG["DevMode"])
    data_not_labeled = BloomFilterDataset(df_not_reidentified_individuals, is_labeled=False, all_two_grams=all_two_grams, dev_mode=GLOBAL_CONFIG["DevMode"])
    bloomfilter_length = len(df_reidentified_individuals["bloomfilter"][0])

if ENC_CONFIG["AliceAlgo"] == "TabMinHash":
    data_labeled = TabMinHashDataset(df_reidentified_individuals, is_labeled=True, all_two_grams=all_two_grams, dev_mode=GLOBAL_CONFIG["DevMode"])
    data_not_labeled = TabMinHashDataset(df_not_reidentified_individuals, is_labeled=False, all_two_grams=all_two_grams, dev_mode=GLOBAL_CONFIG["DevMode"])
    tabminhash_length = len(df_reidentified_individuals["tabminhash"][0])

if (ENC_CONFIG["AliceAlgo"] == "TwoStepHash") & (DEA_CONFIG["TSHMode"] == "Padding"):
    max_length_reidentified = df_reidentified_individuals["twostephash"].apply(lambda x: len(list(x))).max()
    max_length_not_reidentified = df_not_reidentified_individuals["twostephash"].apply(lambda x: len(list(x))).max()
    max_twostephash_length = max(max_length_reidentified, max_length_not_reidentified)
    data_labeled = TwoStepHashDatasetPadding(df_reidentified_individuals, is_labeled=True, all_two_grams=all_two_grams, max_set_size=max_twostephash_length, dev_mode=GLOBAL_CONFIG["DevMode"])
    data_not_labeled = TwoStepHashDatasetPadding(df_not_reidentified_individuals, is_labeled=False, all_two_grams=all_two_grams, max_set_size=max_twostephash_length, dev_mode=GLOBAL_CONFIG["DevMode"])

if (ENC_CONFIG["AliceAlgo"] == "TwoStepHash") & (DEA_CONFIG["TSHMode"] == "FrequencyString"):
    max_length_reidentified = df_reidentified_individuals["twostephash"].apply(lambda x: max(x)).max()
    max_length_not_reidentified = df_not_reidentified_individuals["twostephash"].apply(lambda x: max(x)).max()
    max_twostephash_length = max(max_length_reidentified, max_length_not_reidentified)
    data_labeled = TwoStepHashDatasetFrequencyString(df_reidentified_individuals, is_labeled=True, all_two_grams=all_two_grams, frequency_string_length=max_twostephash_length, dev_mode=GLOBAL_CONFIG["DevMode"])
    data_not_labeled = TwoStepHashDatasetFrequencyString(df_not_reidentified_individuals, is_labeled=False, all_two_grams=all_two_grams, frequency_string_length=max_twostephash_length, dev_mode=GLOBAL_CONFIG["DevMode"])

if (ENC_CONFIG["AliceAlgo"] == "TwoStepHash") & (DEA_CONFIG["TSHMode"] == "OneHotEncoding"):
    unique_integers_reidentified = set().union(*df_reidentified_individuals["twostephash"])
    unique_integers_not_reidentified = set().union(*df_not_reidentified_individuals["twostephash"])
    unique_integers_sorted = sorted(unique_integers_reidentified.union(unique_integers_not_reidentified))
    unique_integers_dict = {i: val for i, val in enumerate(unique_integers_sorted)}
    data_labeled = TwoStepHashDatasetOneHotEncoding(df_reidentified_individuals, is_labeled=True, all_integers=unique_integers_sorted, all_two_grams=all_two_grams, dev_mode=GLOBAL_CONFIG["DevMode"])
    data_not_labeled = TwoStepHashDatasetOneHotEncoding(df_not_reidentified_individuals, is_labeled=False, all_integers=unique_integers_sorted, all_two_grams=all_two_grams, dev_mode=GLOBAL_CONFIG["DevMode"])

# %% [markdown]
# ## Create Dataloader

# %%
# Split proportions
train_size = int(0.8 * len(data_labeled))  # 80% training
val_size = len(data_labeled) - train_size  # 20% validation

#Split dataset of reidentified individuals
data_train, data_val = random_split(data_labeled, [train_size, val_size])

# Create dataloader
dataloader_train = DataLoader(data_train, batch_size=32, shuffle=True)
dataloader_val = DataLoader(data_val, batch_size=32, shuffle=True)
dataloader_test = DataLoader(data_not_labeled, batch_size=1, shuffle=False)

# %% [markdown]
# ## Pytorch Model

# %%
# Create Models based on chosen encoding scheme
if ENC_CONFIG["AliceAlgo"] == "BloomFilter":
    model = BloomFilterToTwoGramClassifier(input_dim=bloomfilter_length, num_two_grams=len(all_two_grams))

if ENC_CONFIG["AliceAlgo"] == "TabMinHash":
    model = TabMinHashToTwoGramClassifier(input_dim=tabminhash_length, num_two_grams=len(all_two_grams))

if (ENC_CONFIG["AliceAlgo"] == "TwoStepHash") & (DEA_CONFIG["TSHMode"] == "Padding"):
    model = TwoStepHashToTwoGramClassifier(input_dim=max_twostephash_length, num_two_grams=len(all_two_grams))

if (ENC_CONFIG["AliceAlgo"] == "TwoStepHash") & (DEA_CONFIG["TSHMode"] == "FrequencyString"):
    model = TabMinHashToTwoGramClassifier(input_dim=max_twostephash_length, num_two_grams=len(all_two_grams))

if (ENC_CONFIG["AliceAlgo"] == "TwoStepHash") & (DEA_CONFIG["TSHMode"] == "OneHotEncoding"):
    model = TabMinHashToTwoGramClassifier(input_dim=len(unique_integers_sorted), num_two_grams=len(all_two_grams))


# %% [markdown]
# ## Training Loop

# %%
# Setup

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function for multi-label classification
criterion = nn.BCEWithLogitsLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%
for data, labels in dataloader_train:
    data, labels = data.to(device), labels.to(device)
    print(data.shape)
    print(labels.shape)
    print(data)
    break

# %%
# Number of epochs
num_epochs = 15
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    for data, labels in tqdm(dataloader_train, desc="Training loop"):
        # Move data to device
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    train_loss = running_loss / len(dataloader_train.dataset)
    train_losses.append(train_loss)

    #Calculate true training loss?

    #Validation
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data, labels in tqdm(dataloader_val, desc="Validation loop"):
            # Move data to device
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
        val_loss = running_loss / len(dataloader_val.dataset)
        val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")

# %% [markdown]
# ## Visualize Losses and Performance Metrics

# %%
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.title("Loss over epochs")
plt.show()

# %% [markdown]
# ## Re-Identify Not-Reidentified Individuals

# %% [markdown]
# ## Visualize Performance for Re-Identification

# %%
sys.exit("Stopping execution at this cell.")