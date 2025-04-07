# %% [markdown]
# # Hyperparameter Optimization

# %% [markdown]
# ## Setup

# %%
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import ray

from utils import *

import matplotlib.pyplot as plt # For data viz
import pandas as pd
import hickle as hkl
import numpy as np
import string
import sys

from ray.air import session

from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

from graphMatching.gma import run_gma

from datasets.bloom_filter_dataset import BloomFilterDataset
from datasets.tab_min_hash_dataset import TabMinHashDataset
from datasets.two_step_hash_dataset_padding import TwoStepHashDatasetPadding
from datasets.two_step_hash_dataset_frequency_string import TwoStepHashDatasetFrequencyString
from datasets.two_step_hash_dataset_one_hot_encoding import TwoStepHashDatasetOneHotEncoding

from pytorch_models_hyperparameter_optimization.base_model import BaseModel

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
    "Overlap": 0.68,
    "DropFrom": "Both",
    "Verbose": True,  # Print Status Messages
    "MatchingMetric": "cosine",
    "Matching": "MinWeight",
    "Workers": -1,
    "SaveAliceEncs": False,
    "SaveEveEncs": False,
    "DevMode": True,
}


DEA_CONFIG = {
    #Padding / FrequencyString / OneHotEncoding
    "TSHMode": "OneHotEncoding",
    "DevMode": False,
    # BCEWithLogitsLoss / MultiLabelSoftMarginLoss
    "LossFunction:": "BCEWithLogitsLoss",
    # Adam / AdamW / SGD / RMSprop
    "Optimizer": "AdamW",
    "LearningRate": 0.001,
    # SGD only
    "Momentum": 0.9,
    "BatchSize": 32,
    "Epochs": 10,
    # TestSize calculated accordingly
    "TrainSize": 0.8,
    "FilterThreshold": 0.5,
    # ReLU / LeakyReLU
    "ActivationFunction": "ReLU",
}

ENC_CONFIG = {
    # TwoStepHash / TabMinHash / BloomFilter
    "AliceAlgo": "BloomFilter",
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

if(os.path.isfile("./data/available_to_eve/reidentified_individuals_%s_%s_%s_%s.h5" % (eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash)) & os.path.isfile("./data/available_to_eve/not_reidentified_individuals_%s_%s_%s_%s.h5" % (eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash))):
    #Load Disk From Data
    reidentified_individuals = hkl.load('./data/available_to_eve/reidentified_individuals_%s_%s_%s_%s.h5' % (eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash))
    df_reidentified_individuals = pd.DataFrame(reidentified_individuals[1:], columns=reidentified_individuals[0])

    not_reidentified_individuals = hkl.load('./data/available_to_eve/not_reidentified_individuals_%s_%s_%s_%s.h5' % (eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash))
    df_not_reidentified_individuals = pd.DataFrame(not_reidentified_individuals[1:], columns=not_reidentified_individuals[0])

    all_individuals = hkl.load('./data/dev/alice_data_complete_with_encoding_%s_%s_%s_%s.h5' % (eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash))
    df_all_individuals = pd.DataFrame(all_individuals[1:], columns=all_individuals[0])

else:
    reidentified_individuals, not_reidentified_individuals, all_individuals = run_gma(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG, DEA_CONFIG, eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash)

    df_reidentified_individuals = pd.DataFrame(reidentified_individuals[1:], columns=reidentified_individuals[0])
    df_not_reidentified_individuals = pd.DataFrame(not_reidentified_individuals[1:], columns=not_reidentified_individuals[0])
    df_all_individuals = pd.DataFrame(all_individuals[1:], columns=all_individuals[0])

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
train_size = int(DEA_CONFIG["TrainSize"] * len(data_labeled))
val_size = len(data_labeled) - train_size

#Split dataset of reidentified individuals
data_train, data_val = random_split(data_labeled, [train_size, val_size])

# Create dataloader
dataloader_train = DataLoader(data_train, batch_size=DEA_CONFIG["BatchSize"], shuffle=True)
dataloader_val = DataLoader(data_val, batch_size=DEA_CONFIG["BatchSize"], shuffle=True)
dataloader_test = DataLoader(data_not_labeled, batch_size=DEA_CONFIG["BatchSize"], shuffle=False)

# %% [markdown]
# ## Pytorch Model

# %% [markdown]
# ## Training Loop

# %%
def train_model(config):
    train_losses, val_losses = [], []

    model = BaseModel(
    input_dim=bloomfilter_length,
    num_two_grams=len(all_two_grams),
    num_layers=config["num_layers"],
    hidden_layer_size=config["hidden_layer_size"],
    dropout_rate=config["dropout_rate"],
    activation_fn=config["activation_fn"]
    )


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define possible loss functions
    loss_functions = {
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss(),
    "MultiLabelSoftMarginLoss": nn.MultiLabelSoftMarginLoss()
    }

    criterion = loss_functions[config["loss_fn"]]

    optimizers = {
        "Adam": optim.Adam(model.parameters(), lr=config["lr"]),
        "SGD": optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9),
        "RMSprop": optim.RMSprop(model.parameters(), lr=config["lr"])
    }

    optimizer = optimizers[config["optimizer"]]

    for epoch in range(config["epochs"]):
        # Training
        model.train()
        running_loss = 0.0
        for data, labels, _ in dataloader_train:
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
            for data, labels, _ in dataloader_val:
                # Move data to device
                data, labels = data.to(device), labels.to(device)

                outputs = model(data)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
            val_loss = running_loss / len(dataloader_val.dataset)
            val_losses.append(val_loss)

    # Switch to evaluation mode
    model.eval()

    # Define Threshhold
    threshold = DEA_CONFIG["FilterThreshold"]

    # Loop through the test dataloader
    with torch.no_grad():  # No need to compute gradients during inference
        for data_batch, uids in dataloader_test:
            # Filter relevant individuals from df_all_individuals
            filtered_df = df_all_individuals[df_all_individuals["uid"].isin(uids)].drop(df_all_individuals.columns[-2], axis=1) # Drop encoding column

            actual_two_grams_batch = []
            for _, entry in filtered_df.iterrows():
                row = entry[:-1] # Exclude UID
                extracted_two_grams = extract_two_grams("".join(map(str, row)))
                actual_two_grams_batch.append({"uid": entry["uid"], "two_grams": extracted_two_grams})

            # Move data to device
            data_batch = data_batch.to(device)

            # Apply model
            logits = model(data_batch)

            # Convert logits to probabilities using sigmoid (for binary classification)
            probabilities = torch.sigmoid(logits)

            # Convert probabilities into 2-gram scores (use two_gram_dict as before)
            batch_two_gram_scores = [
                {two_gram_dict[j]: score.item() for j, score in enumerate(probabilities[i])} #2: For each sample, go through all predicted probabilities (scores)
                for i in range(probabilities.size(0))  # 1: Iterate over each sample in the batch
            ]

            # Apply threshold to filter 2-gram scores (values above threshold are kept)
            batch_filtered_two_gram_scores = [
                {two_gram: score for two_gram, score in two_gram_scores.items() if score > threshold}
                for two_gram_scores in batch_two_gram_scores
            ]

            filtered_two_grams = [
            {"uid": uid, "two_grams": {key for key in two_grams.keys()}}
            for uid, two_grams in zip(uids, batch_filtered_two_gram_scores)
            ]

            sum_dice = 0
            for entry_two_grams_batch in actual_two_grams_batch:  # Loop through each uid in the batch
                for entry_filtered_two_grams in filtered_two_grams:
                    if entry_two_grams_batch["uid"] == entry_filtered_two_grams["uid"]:
                        dice_sim = dice_coefficient(entry_two_grams_batch["two_grams"], entry_filtered_two_grams["two_grams"])
                        sum_dice += dice_sim
    tune.report({"dice": sum(val_losses)})


# %%
# Define extended search space
search_space = {
    "num_layers": tune.randint(1, 8),  # Vary number of layers
    "hidden_layer_size": tune.choice([128, 256, 512, 1024, 2048]),  # Size of hidden layers
    "dropout_rate": tune.uniform(0.1, 0.4),  # Dropout
    "activation_fn": tune.choice(["relu", "leaky_relu", "gelu"]),  # Activation function
    "optimizer": tune.choice(["Adam", "SGD", "RMSprop"]),  # Optimizer selection
    "loss_fn": tune.choice(["BCEWithLogitsLoss"]),  # Loss function selection
    # "loss_fn": tune.choice(["BCEWithLogitsLoss", "MultiLabelSoftMarginLoss"]),  # Loss function selection
    "lr": tune.loguniform(1e-5, 1e-2),  # Learning rate
    "epochs": tune.randint(10, 20),  # Fixed number of epochs
}

# Run Ray Tune with Optuna
ray.init(ignore_reinit_error=True)

optuna_search = OptunaSearch(metric="dice", mode="max")
scheduler = ASHAScheduler(metric="dice", mode="max")

tuner = tune.Tuner(
    train_model,
    tune_config=tune.TuneConfig(
        search_alg=optuna_search,
        scheduler=scheduler,
        num_samples=50  # Number of trials
    ),
    param_space=search_space
)

results = tuner.fit()
print("Best hyperparameters:", results.get_best_result(metric="dice", mode="max").config)

ray.shutdown()


