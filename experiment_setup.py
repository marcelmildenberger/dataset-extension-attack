from main import run_dea
import os
import csv

# === General Parameters ===
GLOBAL_CONFIG = {
    "Data": None,
    "Overlap": None,
    "DropFrom": None,
    "Verbose": False,
    "MatchingMetric": "cosine",
    "Matching": "MinWeight",
    "Workers": 0,
    "SaveAliceEncs": False,
    "SaveEveEncs": False,
    "DevMode": False,
    "BenchMode": True,
    "SaveResults": True,
    "UseGPU": False,
    "SaveModel": False,
    "SavePredictions": False,
}

# === DEA Training Parameters ===
DEA_CONFIG = {
    "TrainSize": 0.8,
    "Patience": 5,
    "MinDelta": 1e-4,
    "NumSamples": 125,
    "Epochs": 20,
    "MetricToOptimize": "average_dice",  # Options: "average_dice", "average_precision", ...
    "MatchingTechnique": "fuzzy_and_greedy",  # Options: "ai", "greedy", "fuzzy", ...
}

# === Encoding Parameters for Alice & Eve ===
ENC_CONFIG = {
    # Encoding technique
    "AliceAlgo": "",
    "AliceSecret": "SuperSecretSalt1337",
    "AliceN": 2,
    "AliceMetric": "dice",
    "EveAlgo": "",
    "EveSecret": "ATotallyDifferentString42",
    "EveN": 2,
    "EveMetric": "dice",

    # Bloom Filter specific
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

    # Tabulation MinHash specific
    "AliceNHash": 1024,
    "AliceNHashBits": 64,
    "AliceNSubKeys": 8,
    "Alice1BitHash": True,
    "EveNHash": 1024,
    "EveNHashBits": 64,
    "EveNSubKeys": 8,
    "Eve1BitHash": True,

    # Two-Step Hashing specific
    "AliceNHashFunc": 10,
    "AliceNHashCol": 1000,
    "AliceRandMode": "PNG",
    "EveNHashFunc": 10,
    "EveNHashCol": 1000,
    "EveRandMode": "PNG",
}

# === Embedding Configuration (e.g., Node2Vec) ===
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
    "EveSeed": 42,
}

# === Graph Alignment Config ===
ALIGN_CONFIG = {
    "RegWS": 0,
    "RegInit": 1,
    "Batchsize": 1,
    "LR": 200.0,
    "NIterWS": 100,
    "NIterInit": 5,
    "NEpochWS": 100,
    "LRDecay": 1,
    "Sqrt": True,
    "EarlyStopping": 10,
    "Selection": "None",
    "MaxLoad": None,
    "Wasserstein": True,
}

# Define the missing experiment combinations (excluding fakename_1k, fakename_2k, titanic_full)
missing_experiments = [
    # Only including experiments for fakename_20k and euro_person
    {"encoding": "TwoStepHash", "dataset": "fakename_20k.tsv", "drop_from": "Both", "overlap": 0.65},
    {"encoding": "TwoStepHash", "dataset": "fakename_20k.tsv", "drop_from": "Eve", "overlap": 0.85},
    {"encoding": "TwoStepHash", "dataset": "fakename_20k.tsv", "drop_from": "Both", "overlap": 0.45},
    {"encoding": "TwoStepHash", "dataset": "fakename_20k.tsv", "drop_from": "Eve", "overlap": 0.65},
    {"encoding": "TwoStepHash", "dataset": "fakename_20k.tsv", "drop_from": "Both", "overlap": 0.25},

    {"encoding": "BloomFilter", "dataset": "fakename_20k.tsv", "drop_from": "Both", "overlap": 0.25},
    {"encoding": "BloomFilter", "dataset": "fakename_20k.tsv", "drop_from": "Both", "overlap": 0.65},
    {"encoding": "BloomFilter", "dataset": "fakename_20k.tsv", "drop_from": "Both", "overlap": 0.45},

    {"encoding": "BloomFilter", "dataset": "euro_person.tsv", "drop_from": "Eve", "overlap": 0.45},
    {"encoding": "BloomFilter", "dataset": "euro_person.tsv", "drop_from": "Both", "overlap": 0.65},
    {"encoding": "BloomFilter", "dataset": "euro_person.tsv", "drop_from": "Eve", "overlap": 0.85},
    {"encoding": "BloomFilter", "dataset": "euro_person.tsv", "drop_from": "Eve", "overlap": 0.25},
    {"encoding": "BloomFilter", "dataset": "euro_person.tsv", "drop_from": "Both", "overlap": 0.45},
    {"encoding": "BloomFilter", "dataset": "euro_person.tsv", "drop_from": "Eve", "overlap": 0.65},
    {"encoding": "BloomFilter", "dataset": "euro_person.tsv", "drop_from": "Both", "overlap": 0.85},
    {"encoding": "BloomFilter", "dataset": "euro_person.tsv", "drop_from": "Both", "overlap": 0.25},
]

print(f"🚀 Starting {len(missing_experiments)} missing experiments...")
print("=" * 80)

for i, exp in enumerate(missing_experiments, 1):
    print(f"\n[{i}/{len(missing_experiments)}] Running: {exp['encoding']} | {exp['dataset']} | {exp['drop_from']} | {exp['overlap']}")

    # Set encoding configuration
    ENC_CONFIG["AliceAlgo"] = exp["encoding"]
    ENC_CONFIG["EveAlgo"] = "None"
    if exp["encoding"] == "BloomFilter":
        ENC_CONFIG["EveAlgo"] = exp["encoding"]

    # Set global configuration
    GLOBAL_CONFIG["Data"] = f"./data/datasets/{exp['dataset']}"
    GLOBAL_CONFIG["DropFrom"] = exp["drop_from"]
    GLOBAL_CONFIG["Overlap"] = exp["overlap"]

    try:
        run_dea(
            GLOBAL_CONFIG.copy(),
            ENC_CONFIG.copy(),
            EMB_CONFIG.copy(),
            ALIGN_CONFIG.copy(),
            DEA_CONFIG.copy()
        )
        print(f"✅ Success: {exp['encoding']} | {exp['dataset']} | {exp['drop_from']} | {exp['overlap']}")
    except Exception as e:
        print(f"❌ Failed: {exp['encoding']} | {exp['dataset']} | {exp['drop_from']} | {exp['overlap']}")
        print(f"   Error: {e}")

print("\n" + "=" * 80)
print("✅ All missing experiments completed!")
print(f"📊 Total experiments run: {len(missing_experiments)}")
