from hyperparameter_optimization import hyperparameter_optimization
# Parameters

# %%
# Parameters
GLOBAL_CONFIG = {
    "Data": "./data/datasets/fakename_10k.tsv",
    "Overlap": 0.8,
    "DropFrom": "Eve",
    "Verbose": False,  # Print Status Messages
    "MatchingMetric": "cosine",
    "Matching": "MinWeight",
    "Workers": -1,
    "SaveAliceEncs": False,
    "SaveEveEncs": False,
    "DevMode": False,
}


DEA_CONFIG = {
    "DevMode": False,
    "BatchSize": 64,
    # TestSize calculated accordingly
    "TrainSize": 0.8,
    "FilterThreshold": 0.5,
    "Patience": 10,
    "MinDelta": 1e-4,
    "NumSamples": 500,
    "Epochs": 20,
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

# Encodings to iterate over
#encs = ["TwoStepHash", "BloomFilter", "TabMinHash"]

#datasets = ["titanic_full.tsv", "fakename_1k.tsv", "fakename_2k.tsv", "fakename_5k.tsv", "fakename_10k.tsv",
#           "fakename_20k.tsv", "fakename_50k.tsv", "euro_full.tsv", "ncvoter.tsv"]
#datasets = ["titanic_full.tsv", "fakename_1k.tsv", "fakename_2k.tsv", "fakename_5k.tsv", "fakename_10k.tsv", "fakename_20k.tsv"]

# drop = ["Alice", "Eve", "Both"]
# Overlaps overlap = [i/100 for i in range(5, 105, 5)]

#for encoding in encs:
#    ENC_CONFIG["AliceAlgo"] = encoding
#    if encoding == 'BloomFilter':
#        ENC_CONFIG["EveAlgo"] = encoding
#    for dataset in datasets:
#        GLOBAL_CONFIG["Data"] = f"./data/datasets/{dataset}"
#        hyperparameter_optimization(GLOBAL_CONFIG.copy(), ENC_CONFIG.copy(), EMB_CONFIG.copy(), ALIGN_CONFIG.copy(), DEA_CONFIG.copy())
hyperparameter_optimization(GLOBAL_CONFIG.copy(), ENC_CONFIG.copy(), EMB_CONFIG.copy(), ALIGN_CONFIG.copy(), DEA_CONFIG.copy())
print("✅ Skript abgeschlossen!")