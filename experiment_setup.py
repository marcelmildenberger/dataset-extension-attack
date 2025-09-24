from main import run_dea
import pandas as pd

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
    "UseGPU": True,
    "SaveModel": False,
    "SavePredictions": False,
    "GraphMatchingAttack": False,
}

# === DEA Training Parameters ===
DEA_CONFIG = {
    "ParallelTrials": 5,
    "HPO": True,
    "TrainSize": 0.8,
    "Patience": 5,
    "MinDelta": 1e-4,
    "NumSamples": 50,
    "Epochs": 25,
    "MetricToOptimize": "average_dice",  # Options: "average_dice", "average_precision", ...
    "MatchingTechnique": "fuzzy_and_greedy",  # Options: "ai", "greedy", "fuzzy", ...,
    "HPO_Narrow_Searchspace": True,
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

# List to store failed experiments
failed_experiments = []

encs = ["TwoStepHash"]

datasets = ["fakename_20k.tsv", "fakename_50k.tsv", "euro_person.tsv"]

overlap = [0.6]

for dataset in datasets:
    for encoding in encs:
        ENC_CONFIG["AliceAlgo"] = encoding
        ENC_CONFIG["EveAlgo"] = "None"
        if encoding == "BloomFilter":
            ENC_CONFIG["EveAlgo"] = encoding
        for ov in overlap:
            GLOBAL_CONFIG["Data"] = f"./data/datasets/{dataset}"
            GLOBAL_CONFIG["Overlap"] = ov
            try:
                run_dea(
                    GLOBAL_CONFIG.copy(),
                    ENC_CONFIG.copy(),
                    EMB_CONFIG.copy(),
                    ALIGN_CONFIG.copy(),
                    DEA_CONFIG.copy()
                )
            except Exception as e:
                # Record failed experiment
                failed_experiments.append({
                    "encoding": encoding,
                    "dataset": dataset,
                    "overlap": ov,
                    "error_message": str(e),
                    "error_type": type(e).__name__,
                })
                print(f"Failed: {encoding} - {dataset} - {ov}: {e}")
                continue

# Save failed experiments to CSV
if failed_experiments:
    failed_df = pd.DataFrame(failed_experiments)
    failed_df.to_csv("experiment_results/failed_experiments.csv", index=False)
    print(f"\nSaved {len(failed_experiments)} failed experiments to failed_experiments.csv")
else:
    print("\nNo failed experiments to save.")