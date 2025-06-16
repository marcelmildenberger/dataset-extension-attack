# %% [markdown]
# # Privacy-Preserving Record Linkage (PPRL): Investigating Dataset Extension Attacks

# %% [markdown]
# ## Preparation

# %% [markdown]
# ### Imports
#
# Import all relevant libraries and classes used throughout the project. Key components include:
#
# - **Torch** – for tensor operations and neural network functionality
# - **Datasets** – for handling training and evaluation data
# - **PyTorch Models** – custom and pre-defined models for the DEA
# - **Graph Matching Attack (GMA)** – core logic for the initial re-identification phase
#

# %%
import os
import json
from datetime import datetime
import seaborn as sns

from functools import partial  # Import partial from functools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset

import ray
from ray import tune, air
from ray import train
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler


from torch.utils.tensorboard import SummaryWriter

from tqdm.notebook import tqdm

import numpy as np




from utils import get_hashes, convert_to_two_gram_scores, filter_two_grams, calculate_performance_metrics, run_epoch, label_tensors_to_two_grams, reconstruct_using_ai, reidentification_analysis, clean_result_dict, print_and_save_result, resolve_config, two_gram_overlap, reconstruct_using_ai_from_reconstructed_strings, lowercase_df, create_identifier_column, find_most_likely_birthday, find_most_likely_given_name, find_most_likely_surnames, longest_path_reconstruction

import matplotlib.pyplot as plt # For data viz
import pandas as pd
import hickle as hkl
import string
from early_stopping.early_stopping import EarlyStopping

from graphMatching.gma import run_gma

from datasets.bloom_filter_dataset import BloomFilterDataset
from datasets.tab_min_hash_dataset import TabMinHashDataset
from datasets.two_step_hash_dataset import TwoStepHashDataset

from pytorch_models_hyperparameter_optimization.base_model_hyperparameter_optimization import BaseModelHyperparameterOptimization
from pytorch_models.base_model import BaseModel

def run_dea(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG, DEA_CONFIG):

    # %%
    # --- Generate a dictionary of all possible 2-grams from letters and digits ---

    # Lowercase alphabet: 'a' to 'z'
    alphabet = string.ascii_lowercase

    # Digits: '0' to '9'
    digits = string.digits

    # Generate all letter-letter 2-grams (e.g., 'aa', 'ab', ..., 'zz')
    letter_letter_grams = [a + b for a in alphabet for b in alphabet]

    # Generate all digit-digit 2-grams (e.g., '00', '01', ..., '99')
    digit_digit_grams = [d1 + d2 for d1 in digits for d2 in digits]

    # Generate all letter-digit 2-grams (e.g., 'a0', 'a1', ..., 'z9')
    letter_digit_grams = [l + d for l in alphabet for d in digits]

    # Combine all generated 2-grams into one list
    all_two_grams = letter_letter_grams + letter_digit_grams + digit_digit_grams

    # Create a dictionary mapping index to each 2-gram
    two_gram_dict = {i: two_gram for i, two_gram in enumerate(all_two_grams)}

    # %% [markdown]
    # ## Step 1: Load or Compute Graph Matching Attack (GMA) Results
    #
    # This code snippet either loads previously computed Graph Matching Attack (GMA) results from disk or runs the attack if no saved data is found.
    #
    # 1. **Generate Configuration Hashes:**
    #    The function `get_hashes` creates unique hash values based on the encoding and embedding configurations. These are used to create distinct filenames for the data.
    #
    # 2. **Create File Paths:**
    #    Based on the configuration hashes, paths are generated for:
    #    - Reidentified individuals
    #    - Not reidentified individuals
    #    - All individuals in Alice’s dataset (with encoding)
    #
    # 3. **Load Results from Disk (if available):**
    #    If the `.h5` files already exist, they are loaded using `hickle` and converted into `pandas.DataFrames`.
    #    The data format assumes that the first row contains the column headers, and the rest is the data — hence the slicing `[1:]` and `columns=...`.
    #
    # 4. **Run GMA If Data Is Not Available:**
    #    If the files are missing, the GMA is executed via `run_gma()`. The results are again converted to `DataFrames`.
    #

    # %%
    data_dir = os.path.abspath("./data")

    eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash = get_hashes(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG)

    identifier = f"{eve_enc_hash}_{alice_enc_hash}_{eve_emb_hash}_{alice_emb_hash}"

    # Define file paths based on the configuration hashes
    path_reidentified = f"{data_dir}/available_to_eve/reidentified_individuals_{identifier}.h5"
    path_not_reidentified = f"{data_dir}/available_to_eve/not_reidentified_individuals_{identifier}.h5"
    path_all = f"{data_dir}/dev/alice_data_complete_with_encoding_{identifier}.h5"

    # Check if the output files already exist
    if not (os.path.isfile(path_reidentified) and os.path.isfile(path_all) and os.path.isfile(path_not_reidentified)):
        run_gma(
            GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG, DEA_CONFIG,
            eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash
        )

    # %% [markdown]
    # ## 🧩 Step 2: Data Preparation
    #
    # This section initializes the dataset objects depending on which encoding method Alice used. Each encoding requires a different preprocessing strategy for compatibility with downstream neural models.
    #
    # ### 1. Bloom Filter (`"BloomFilter"`)
    # - Uses binary Bloom filters to represent identifiers.
    # - Loads `BloomFilterDataset` objects.
    # - Stores the bit-length of the bloom filter.
    #
    # ### 2. Tabulation MinHash (`"TabMinHash"`)
    # - Applies a MinHash-based encoding.
    # - Loads `TabMinHashDataset`.
    # - Captures the length of each encoded vector.
    #
    # ### 3. Two-Step Hash with One-Hot Encoding (`"TwoStepHash"`)
    # - Extracts all **unique hash values** to build a consistent one-hot vector space.
    # - Constructs datasets using `TwoStepHashDatasetOneHotEncoding`.
    #
    # > ⚙️ All dataset constructors are passed:
    # > - Whether the data is labeled
    # > - The full 2-gram list (used as feature tokens)
    # > - Additional encoding-specific configurations
    # > - Dev mode toggle (for debugging or smaller runs)
    #

    # %%
    def load_data(data_directory, identifier, load_test=False):
        # Get unique hash identifiers for the encoding and embedding configurations

        data_train = data_val = data_test = None

        if load_test:
            path_not_reidentified = f"{data_directory}/available_to_eve/not_reidentified_individuals_{identifier}.h5"
            path_all = f"{data_directory}/dev/alice_data_complete_with_encoding_{identifier}.h5"
            not_reidentified_data = hkl.load(path_not_reidentified)
            all_data = hkl.load(path_all)
            # Convert lists to DataFrames
            df_not_reidentified = pd.DataFrame(not_reidentified_data[1:], columns=not_reidentified_data[0])
            df_all = pd.DataFrame(all_data[1:], columns=all_data[0])
            df_not_reidentified_labeled = df_all[df_all["uid"].isin(df_not_reidentified["uid"])].reset_index(drop=True)

        # Define file paths based on the configuration hashes
        path_reidentified = f"{data_directory}/available_to_eve/reidentified_individuals_{identifier}.h5"

        reidentified_data = hkl.load(path_reidentified)

        # Convert lists to DataFrames
        df_reidentified = pd.DataFrame(reidentified_data[1:], columns=reidentified_data[0])

        # 1️⃣ Bloom Filter Encoding
        if ENC_CONFIG["AliceAlgo"] == "BloomFilter":
            data_labeled = BloomFilterDataset(
                df_reidentified,
                is_labeled=True,
                all_two_grams=all_two_grams,
                dev_mode=GLOBAL_CONFIG["DevMode"]
            )
            if load_test:
                # If loading validation data, also create a dataset for not reidentified individuals
                data_test = BloomFilterDataset(
                    df_not_reidentified_labeled,
                    is_labeled=True,
                    all_two_grams=all_two_grams,
                    dev_mode=GLOBAL_CONFIG["DevMode"]
                )

        # 2️⃣ Tabulation MinHash Encoding
        elif ENC_CONFIG["AliceAlgo"] == "TabMinHash":
            data_labeled = TabMinHashDataset(
                df_reidentified,
                is_labeled=True,
                all_two_grams=all_two_grams,
                dev_mode=GLOBAL_CONFIG["DevMode"]
            )
            if load_test:
                # If loading validation data, also create a dataset for not reidentified individuals
                data_test = TabMinHashDataset(
                    df_not_reidentified_labeled,
                    is_labeled=True,
                    all_two_grams=all_two_grams,
                    dev_mode=GLOBAL_CONFIG["DevMode"]
                )

        # 3 Two-Step Hash Encoding (One-Hot Encoding Mode)
        elif ENC_CONFIG["AliceAlgo"] == "TwoStepHash":
            # Collect all unique integers across both reidentified and non-reidentified data
            unique_ints = sorted(set().union(*df_reidentified["twostephash"]))
            data_labeled = TwoStepHashDataset(
                df_reidentified,
                is_labeled=True,
                all_integers=unique_ints,
                all_two_grams=all_two_grams,
                dev_mode=GLOBAL_CONFIG["DevMode"]
            )
            if load_test:
                # If loading validation data, also create a dataset for not reidentified individuals
                data_test = TwoStepHashDataset(
                    df_not_reidentified_labeled,
                    is_labeled=True,
                    all_integers=unique_ints,
                    all_two_grams=all_two_grams,
                    dev_mode=GLOBAL_CONFIG["DevMode"]
                )

        # Define dataset split proportions
        train_size = int(DEA_CONFIG["TrainSize"] * len(data_labeled))
        val_size = len(data_labeled) - train_size

        # Split the reidentified dataset into training and validation sets
        data_train, data_val = random_split(data_labeled, [train_size, val_size])

        return data_train, data_val, data_test

    def load_not_reidentified_data(data_directory, identifier):
        path_not_reidentified = f"{data_directory}/available_to_eve/not_reidentified_individuals_{identifier}.h5"
        path_all = f"{data_directory}/dev/alice_data_complete_with_encoding_{identifier}.h5"
        not_reidentified_data = hkl.load(path_not_reidentified)
        all_data = hkl.load(path_all)
        # Convert lists to DataFrames
        df_not_reidentified = pd.DataFrame(not_reidentified_data[1:], columns=not_reidentified_data[0])
        df_all = pd.DataFrame(all_data[1:], columns=all_data[0])
        df_not_reidentified_labeled = df_all[df_all["uid"].isin(df_not_reidentified["uid"])].reset_index(drop=True).drop("bloomfilter", axis=1)

        return df_not_reidentified_labeled

    # %% [markdown]
    # ## Step 3: Hyperparameter Optimization

    # %%
    def train_model(config, data_dir, output_dim, identifier, patience, min_delta):
        # Create DataLoaders for training, validation, and testing

        data_train, data_val, _ = load_data(data_dir, identifier, load_test=False)

        input_dim = data_train[0][0].shape[0]  # Get the input dimension from the first sample

        dataloader_train = DataLoader(
            data_train,
            batch_size=int(config["batch_size"]),
            shuffle=True  # Important for training
        )

        dataloader_val = DataLoader(
            data_val,
            batch_size=int(config["batch_size"]),
            shuffle=True  # Allows variation in validation batches
        )

        train_losses = []
        val_losses = []
        total_precision = total_recall = total_f1 = total_dice = total_val_loss = 0.0
        n = len(dataloader_val.dataset)
        epochs = 0
        early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)

        # Define and initialize model with hyperparameters from config
        model = BaseModelHyperparameterOptimization(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=config["num_layers"],
            hidden_layer_size=config["hidden_layer_size"],
            dropout_rate=config["dropout_rate"],
            activation_fn=config["activation_fn"]
        )

        # Set device for model (GPU or CPU)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Select loss function based on config
        loss_functions = {
            "BCEWithLogitsLoss": nn.BCEWithLogitsLoss(),
            "MultiLabelSoftMarginLoss": nn.MultiLabelSoftMarginLoss(),
            "SoftMarginLoss": nn.SoftMarginLoss(),
        }
        criterion = loss_functions[config["loss_fn"]]

        learning_rate = config["optimizer"]["lr"].sample()
        # Select optimizer based on config
        optimizers = {
            "Adam": lambda: optim.Adam(model.parameters(), lr=learning_rate),
            "AdamW": lambda: optim.AdamW(model.parameters(), lr=learning_rate),
            "SGD": lambda: optim.SGD(model.parameters(), lr=learning_rate, momentum=config["optimizer"]["momentum"].sample()),
            "RMSprop": lambda: optim.RMSprop(model.parameters(), lr=learning_rate)
        }
        optimizer = optimizers[config["optimizer"]["name"]]()

        schedulers = {
            "StepLR": lambda: torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config["lr_scheduler"]["step_size"].sample(),
                gamma=config["lr_scheduler"]["gamma"].sample()
            ),
            "ExponentialLR": lambda: torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=config["lr_scheduler"]["gamma"].sample()
            ),
            "ReduceLROnPlateau": lambda: torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=config["lr_scheduler"]["mode"],
                factor=config["lr_scheduler"]["factor"].sample(),
                patience=config["lr_scheduler"]["patience"].sample()
            ),
            "CosineAnnealingLR": lambda: torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config["lr_scheduler"]["T_max"].sample(),
                eta_min=config["lr_scheduler"]["eta_min"].sample()
            ),
            "CyclicLR": lambda: torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=config["lr_scheduler"]["base_lr"].sample(),
                max_lr=config["lr_scheduler"]["max_lr"].sample(),
                step_size_up=config["lr_scheduler"]["step_size_up"].sample(),
                mode=config["lr_scheduler"]["mode_cyclic"].sample(),
                cycle_momentum=False
            ),
            "None": lambda: None,
        }
        scheduler = schedulers[config["lr_scheduler"]["name"]]()

        # Training loop
        for _ in range(DEA_CONFIG["Epochs"]):
            epochs += 1
            # Training phase
            model.train()
            train_loss = run_epoch(model, dataloader_train, criterion, optimizer, device, is_training=True, verbose=GLOBAL_CONFIG["Verbose"], scheduler=scheduler)
            train_losses.append(train_loss)

            # Validation phase
            model.eval()
            val_loss = run_epoch(model, dataloader_val, criterion, optimizer, device, is_training=False, verbose=GLOBAL_CONFIG["Verbose"], scheduler=scheduler)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            val_losses.append(val_loss)
            total_val_loss += val_loss

            # Early stopping check
            if early_stopper(val_loss):
                break

        # Test phase with reconstruction and evaluation
        model.eval()

        with torch.no_grad():
            for data, labels, _ in dataloader_val:

                actual_two_grams = label_tensors_to_two_grams(two_gram_dict, labels)

                # Move data to device and make predictions
                data = data.to(device)
                logits = model(data)
                probabilities = torch.sigmoid(logits)

                # Convert probabilities into 2-gram scores
                batch_two_gram_scores = convert_to_two_gram_scores(two_gram_dict, probabilities)

                # Filter out low-scoring 2-grams
                batch_filtered_two_gram_scores = filter_two_grams(batch_two_gram_scores, config["threshold"])

                # Calculate performance metrics for evaluation
                dice, precision, recall, f1 = calculate_performance_metrics(
                    actual_two_grams, batch_filtered_two_gram_scores)

                total_dice += dice
                total_precision += precision
                total_recall += recall
                total_f1 += f1

        train.report({
                "average_dice": total_dice / n,
                "average_precision": total_precision / n,
                "average_recall": total_recall / n,
                "average_f1": total_f1 / n,
                "total_val_loss": total_val_loss,
                "len_train": len(dataloader_train.dataset),
                "len_val": len(dataloader_val.dataset),
                "epochs": epochs
            })

    # %%
    # Define search space for hyperparameter optimization
    search_space = {
        "output_dim": len(all_two_grams),  # Output dimension is also the number of unique 2-grams
        "num_layers": tune.randint(1, 6),  # Vary the number of layers in the model
        #"num_layers": tune.randint(1, 2),
        "hidden_layer_size": tune.choice([64, 128, 256, 512, 1024, 2048, 4096]),  # Different sizes for hidden layers
        #"hidden_layer_size": tune.choice([1024, 2048]),  # Different sizes for hidden layers
        "dropout_rate": tune.uniform(0.1, 0.4),  # Dropout rate between 0.1 and 0.4
        "activation_fn": tune.choice(["relu", "leaky_relu", "gelu", "elu", "selu", "tanh"]),  # Activation functions to choose from
        "optimizer": tune.choice([
            {"name": "Adam", "lr": tune.loguniform(1e-5, 1e-3)},
            {"name": "AdamW", "lr": tune.loguniform(1e-5, 1e-3)},
            {"name": "SGD", "lr": tune.loguniform(1e-4, 1e-2), "momentum": tune.uniform(0.0, 0.99)},
            {"name": "RMSprop", "lr": tune.loguniform(1e-5, 1e-3)},
        ]),
        "loss_fn": tune.choice(["BCEWithLogitsLoss", "MultiLabelSoftMarginLoss", "SoftMarginLoss"]),
        "threshold": tune.uniform(0.3, 0.8),
        "lr_scheduler": tune.choice([
            {"name": "StepLR", "step_size": tune.choice([5, 10, 20]), "gamma": tune.uniform(0.1, 0.9)},
            {"name": "ExponentialLR", "gamma": tune.uniform(0.85, 0.99)},
            {"name": "ReduceLROnPlateau", "mode": "min", "factor": tune.uniform(0.1, 0.5), "patience": tune.choice([5, 10, 15])},
            {"name": "CosineAnnealingLR", "T_max": tune.loguniform(10, 50) , "eta_min": tune.choice([1e-5, 1e-6, 0])},
            {"name": "CyclicLR", "base_lr": tune.loguniform(1e-5, 1e-3), "max_lr": tune.loguniform(1e-3, 1e-1), "step_size_up": tune.choice([2000, 4000]), "mode_cyclic": tune.choice(["triangular", "triangular2", "exp_range"]) },
            {"name": "None"}  # No scheduler
        ]),
        "batch_size": tune.choice([8, 16, 32, 64]),  # Batch sizes to test
    }

    selected_dataset = GLOBAL_CONFIG["Data"].split("/")[-1].replace(".tsv", "")

    experiment_tag = "experiment_" + ENC_CONFIG["AliceAlgo"] + "_" + selected_dataset + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Initialize Ray for hyperparameter optimization
    ray.init(ignore_reinit_error=True)

    # Optuna Search Algorithm for optimizing the hyperparameters
    optuna_search = OptunaSearch(metric=DEA_CONFIG["MetricToOptimize"], mode="max")

    # Use ASHAScheduler to manage trials and early stopping
    scheduler = ASHAScheduler(metric="total_val_loss", mode="min")

    # Define and configure the Tuner for Ray Tune
    tuner = tune.Tuner(
        partial(train_model, data_dir=data_dir, output_dim=len(all_two_grams), identifier=identifier , patience=DEA_CONFIG["Patience"], min_delta=DEA_CONFIG["MinDelta"]),  # The function to optimize (training function)
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,  # Search strategy using Optuna
            scheduler=scheduler,  # Use ASHA to manage the trials
            num_samples=DEA_CONFIG["NumSamples"],  # Number of trials to run
            max_concurrent_trials=DEA_CONFIG["NumCPU"],
        ),
        param_space=search_space  # Pass in the defined hyperparameter search space
    )

    # Run the tuner
    results = tuner.fit()

    # Shut down Ray after finishing the optimization
    ray.shutdown()

    # %%
    result_grid = results
    best_result = result_grid.get_best_result(metric="average_dice", mode="max")

    # %%
    if DEA_CONFIG["SaveResults"]:
        save_to = f"experiment_results/{experiment_tag}"
        os.makedirs(save_to, exist_ok=True)
        os.makedirs(f"{save_to}/hyperparameteroptimization", exist_ok=True)
        worst_result = result_grid.get_best_result(metric="average_dice", mode="min")

        # Combine configs and metrics into a DataFrame
        df = pd.DataFrame([
            {
                **clean_result_dict(resolve_config(result.config)),
                **{k: result.metrics.get(k) for k in ["average_dice", "average_precision", "average_recall", "average_f1"]},
            }
            for result in result_grid
        ])

        # Save to CSV
        df.to_csv(f"{save_to}/hyperparameteroptimization/all_trial_results.csv", index=False)
        print("✅ Results saved to all_trial_results.csv")

        print_and_save_result("Best_Result", best_result, f"{save_to}/hyperparameteroptimization")
        print_and_save_result("Worst_Result", worst_result, f"{save_to}/hyperparameteroptimization")

        # Compute and print average metrics
        print("\n📊 Average Metrics Across All Trials")
        avg_metrics = df[["average_dice", "average_precision", "average_recall", "average_f1"]].mean()
        print("-" * 40)
        for key, value in avg_metrics.items():
            print(f"{key.capitalize()}: {value:.4f}")

        # --- 📈 Plotting performance metrics ---
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[["average_dice", "average_recall", "average_f1", "average_precision"]])
        plt.title("Distribution of Performance Metrics Across Trials")
        plt.grid(True)
        plt.savefig(f"{save_to}/hyperparameteroptimization/metric_distributions.png")
        print("📊 Saved plot: metric_distributions.png")

        # --- 📌 Correlation between config params and performance ---
        # Only include numeric config columns
        exclude_cols = {"input_dim", "output_dim"}
        numeric_config_cols = [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col]) and col not in exclude_cols
        ]
        correlation_df = df[numeric_config_cols].corr()

        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_df, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Between Parameters and Metrics")
        plt.tight_layout()
        plt.savefig(f"{save_to}/hyperparameteroptimization/correlation_heatmap.png")
        print("📌 Saved heatmap: correlation_heatmap.png")


    # %% [markdown]
    # ## Step 4: Model Training
    #
    # The neural network model is selected dynamically based on the encoding technique used for Alice’s data.
    #
    # ### Supported Models:
    #
    # - **BloomFilter** → `BloomFilterToTwoGramClassifier`
    #   - Input: Binary vector (Bloom filter)
    #   - Output: 2-gram prediction
    #
    # - **TabMinHash** → `TabMinHashToTwoGramClassifier`
    #   - Input: Tabulated MinHash signature
    #   - Output: 2-gram prediction
    #
    # - **TwoStepHash** → `TwoStepHashToTwoGramClassifier`
    #   - Input: Length of the unique integers present
    #   - Output: 2-gram predicition
    #
    # Each model outputs predictions over the set of all possible 2-grams (`all_two_grams`), and the input dimension is dynamically configured based on the dataset.
    #

    # %%
    best_config = resolve_config(best_result.config)
    data_train, data_val, data_test = load_data(data_dir, identifier, load_test=True)
    input_dim=data_train[0][0].shape[0]

    dataloader_train = DataLoader(
        data_train,
        batch_size=int(best_config.get("batch_size", 32)),  # Default to 32 if not specified
        shuffle=True  # Important for training
    )

    dataloader_val = DataLoader(
        data_val,
        batch_size=int(best_config.get("batch_size", 32)),
        shuffle=True  # Allows variation in validation batches
    )

    dataloader_test = DataLoader(
        data_test,
        batch_size=int(best_config.get("batch_size", 32)),
        shuffle=True  # Allows variation in validation batches
    )

    # %%
    model = BaseModel(
                input_dim=input_dim,
                output_dim=len(all_two_grams),
                hidden_layer=best_config.get("hidden_layer_size", 128),  # Default to 128 if not specified
                num_layers=best_config.get("num_layers", 2),  # Default to 2 if not specified
                dropout_rate=best_config.get("dropout_rate", 0.2),  # Default to 0.2 if not specified
                activation_fn=best_config.get("activation_fn", "relu")  # Default to 'relu' if not specified
            )
    print(model)

    # %% [markdown]
    # ## Training Environment Setup
    # This code initializes the core components needed for training a neural network model.
    #
    # 1. TensorBoard Setup
    #     - Creates unique run name by combining:
    #     - Loss function type
    #     - Optimizer choice
    #     - Alice's algorithm
    #     - Initializes TensorBoard writer in runs directory
    # 2. Device Configuration
    #     - Automatically selects GPU if available, falls back to CPU
    #     - Moves model to selected device
    # 3. Loss Functions
    #     - `BCEWithLogitsLoss`: Binary Cross Entropy with Logits
    #     - `MultiLabelSoftMarginLoss`: Multi-Label Soft Margin Loss
    # 4. Optimizers:
    #     - `Adam`: Adaptive Moment Estimation
    #     - `AdamW`: Adam with Weight Decay
    #     - `SGD`: Stochastic Gradient Descent (with momentum)
    #     - `RMSprop`: Root Mean Square Propagation

    # %%
    if DEA_CONFIG["SaveResults"]:
        # Setup tensorboard logging
        run_name = "".join([
            best_config.get("loss_fn", "MultiLabelSoftMarginLoss"),
            best_config.get("optimizer").get("name", "Adam"),
            ENC_CONFIG["AliceAlgo"],
            best_config.get("activation_fn", "relu"),
        ])
        tb_writer = SummaryWriter(f"{save_to}/{run_name}")

    # Setup compute device (GPU/CPU)
    compute_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(compute_device)

    # Initialize loss function
    match best_config.get("loss_fn", "MultiLabelSoftMarginLoss"):
        case "BCEWithLogitsLoss":
            criterion = nn.BCEWithLogitsLoss(reduction='mean')
        case "MultiLabelSoftMarginLoss":
            criterion = nn.MultiLabelSoftMarginLoss(reduction='mean')
        case "SoftMarginLoss":
            criterion = nn.SoftMarginLoss()
        case _:
            raise ValueError(f"Unsupported loss function: {best_config.get('loss_fn', 'MultiLabelSoftMarginLoss')}")

    # Initialize optimizer
    match best_config.get("optimizer").get("name", "Adam"):
        case "Adam":
            optimizer = optim.Adam(model.parameters(), lr=best_config.get("optimizer").get("lr"))
        case "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=best_config.get("optimizer").get("lr"))
        case "SGD":
            optimizer = optim.SGD(model.parameters(),
                                lr=best_config.get("optimizer").get("lr"),
                                momentum=best_config.get("optimizer").get("momentum"))
        case "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=best_config.get("optimizer").get("lr"))
        case _:
            raise ValueError(f"Unsupported optimizer: {best_config.get('optimizer').get('name', 'Adam')}")

    # Initialize learning rate scheduler
    match best_config.get("lr_scheduler").get("name", "None"):
        case "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=best_config.get("lr_scheduler").get("step_size"),
                gamma=best_config.get("lr_scheduler").get("gamma")
            )
        case "ExponentialLR":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=best_config.get("lr_scheduler").get("gamma")
            )
        case "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=best_config.get("lr_scheduler").get("mode"),
                factor=best_config.get("lr_scheduler").get("factor"),
                patience=best_config.get("lr_scheduler").get("patience")
            )
        case "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=best_config.get("lr_scheduler").get("T_max")
            )
        case "CyclicLR":
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=best_config.get("lr_scheduler").get("base_lr"),
                max_lr=best_config.get("lr_scheduler").get("max_lr"),
                step_size_up=best_config.get("lr_scheduler").get("step_size_up"),
                mode=best_config.get("lr_scheduler").get("mode_cyclic"),
                cycle_momentum=False  # usually False for Adam/AdamW
            )
        case None | "None":
            scheduler = None
        case _:
            raise ValueError(f"Unsupported LR scheduler: {best_config.get('lr_scheduler').get('name', 'None')}")

    # %% [markdown]
    # ## Model Training with Early Stopping
    #
    # The function `train_model` orchestrates the training process for the neural network, including both training and validation phases for each epoch. It also utilizes **early stopping** to halt training when the validation loss fails to improve over multiple epochs, avoiding overfitting.
    #
    # ### Key Phases:
    # 1. **Training Phase**:
    #    - The model is trained on the `dataloader_train`, computing the training loss using the specified loss function (`criterion`) and optimizer. Gradients are calculated, and the model parameters are updated.
    #
    # 2. **Validation Phase**:
    #    - The model is evaluated on the `dataloader_val` without updating weights. The validation loss is computed to track model performance on unseen data.
    #
    # 3. **Logging**:
    #    - Training and validation losses are logged to both the console and **TensorBoard** for tracking model performance during training.
    #
    # 4. **Early Stopping**:
    #    - If the validation loss does not improve after a certain number of epochs (defined by `DEA_CONFIG["Patience"]`), the training process is halted to prevent overfitting.
    #
    # ### Helper Functions:
    # - `run_epoch`: Handles a single epoch, either for training or validation, depending on the flag `is_training`.
    # - `log_metrics`: Logs the training and validation losses to the console and TensorBoard for each epoch.
    #

    # %%
    def train_model(model, dataloader_train, dataloader_val, criterion, optimizer, device):
        train_losses = []
        val_losses = []
        val_loss = train_loss = 0.0
        early_stopper = EarlyStopping(patience=DEA_CONFIG["EarlyStoppingPatience"], min_delta=DEA_CONFIG["MinDelta"], verbose=GLOBAL_CONFIG["Verbose"])

        for epoch in range(best_config.get("epochs", DEA_CONFIG["Epochs"])):  # Use best_config epochs or default to DEA_CONFIG
            # Training phase
            model.train()
            train_loss = run_epoch(
                model, dataloader_train, criterion, optimizer,
                device, is_training=True, verbose=GLOBAL_CONFIG["Verbose"], scheduler=scheduler
            )
            train_losses.append(train_loss)

            # Validation phase
            model.eval()
            val_loss = run_epoch(
                model, dataloader_val, criterion, optimizer,
                device, is_training=False, verbose=GLOBAL_CONFIG["Verbose"], scheduler=scheduler
            )
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            val_losses.append(val_loss)

            # Logging
            log_metrics(train_loss, val_loss, epoch, best_config.get("epochs", DEA_CONFIG["Epochs"]))

            # Early stopping check
            if early_stopper(val_loss):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        return train_losses, val_losses

    def log_metrics(train_loss, val_loss, epoch, total_epochs):
        print(f"Epoch {epoch + 1}/{total_epochs} - "
            f"Train loss: {train_loss:.4f}, "
            f"Validation loss: {val_loss:.4f}")
        if DEA_CONFIG["SaveResults"]:
            tb_writer.add_scalar("Loss/train", train_loss, epoch + 1)
            tb_writer.add_scalar("Loss/validation", val_loss, epoch + 1)

    train_losses, val_losses = train_model(
        model, dataloader_train, dataloader_val,
        criterion, optimizer, compute_device
        )

    # %% [markdown]
    # ## Loss Visualization over Epochs
    #
    # This code snippet generates a plot to visualize the **training loss** and **validation loss** across epochs. It's useful for tracking model performance during training and evaluating if overfitting is occurring (i.e., when validation loss starts increasing while training loss continues to decrease).
    #
    # ### Key Elements:
    # 1. **Plotting the Losses**:
    #    - The `train_losses` and `val_losses` are plotted over the epochs.
    #    - The **blue line** represents the training loss, and the **red line** represents the validation loss.
    #
    # 2. **Legend**:
    #    - A legend is added to distinguish between training and validation losses.
    #
    # 3. **Title and Labels**:
    #    - The plot is titled "Training and Validation Loss over Epochs" for context.
    #    - **X-axis** represents the epoch number, and **Y-axis** represents the loss value.
    #

    # %%
    # Plot the training and validation losses over epochs
    plt.plot(train_losses, label='Training loss', color='blue')
    plt.plot(val_losses, label='Validation loss', color='red')

    # Adding a legend to the plot
    plt.legend()

    # Setting the title and labels for clarity
    plt.title("Training and Validation Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    # Show the plot
    plt.show()

    # %% [markdown]
    # ## Step 5: Application to Encoded Data
    #
    # This code performs inference on the test data and compares the predicted 2-grams with the actual 2-grams, providing a performance evaluation based on the **Dice similarity coefficient**.
    #
    # ### Key Steps:
    #
    # 1. **Prepare for Evaluation**:
    #    - The model is switched to **evaluation mode** (`model.eval()`), ensuring no gradient computation.
    #
    # 2. **Thresholding**:
    #    - A threshold (`DEA_CONFIG["FilterThreshold"]`) is applied to filter out low-probability predictions, retaining only the most confident predictions.
    #
    # 3. **Inference and 2-Gram Scoring**:
    #    - The model is applied to the batch, and the **logits** are converted into probabilities using the **sigmoid function**.
    #    - The probabilities are then mapped to **2-gram scores**, and scores below the threshold are discarded.
    #
    # 4. **Reconstructing Words**:
    #    - For each sample in the batch, **2-grams** are reconstructed into words based on the filtered scores.
    #
    # 5. **Performance Metrics**:
    #    - The actual 2-grams (from the test dataset) are compared with the predicted 2-grams, and the **Dice similarity coefficient** is calculated for each sample.
    #
    # ### Result:
    # - The code generates a list `combined_results_performance`, which contains a detailed comparison for each UID, including:
    #   - **Actual 2-grams** (from the test data)
    #   - **Predicted 2-grams** (from the model)
    #   - **Dice similarity** score indicating how similar the actual and predicted 2-grams are.

    # %%
    #save_to = "experiment_results/experiment_BloomFilter_fakename_5k_2025-06-06_14-19-00" # overwrite if necessary
    base_path = DEA_CONFIG["LoadPath"] if DEA_CONFIG["LoadResults"] else save_to
    base_path = os.path.join(base_path, "trained_model")
    model_file = os.path.join(base_path, "model.pt")
    config_file = os.path.join(base_path, "config.json")
    result_file = os.path.join(base_path, "result.json")
    metrics_file = os.path.join(base_path, "metrics.txt")

    # %%
    if DEA_CONFIG["SaveResults"]:
        print(f"Saving results to {save_to}")
        os.makedirs(base_path, exist_ok=True)
        torch.save(model.state_dict(), model_file)
        with open(config_file, "w") as f:
            json.dump(best_config, f, indent=4)

    # %%
    if DEA_CONFIG["LoadResults"]:
        with open(config_file) as f:
            best_config = json.load(f)

        model = BaseModel(
            input_dim=1024,
            output_dim=1036,
            hidden_layer=best_config.get("hidden_layer_size", 128),  # Default to 128 if not specified
            num_layers=best_config.get("num_layers", 2),  # Default to 2 if not specified
            dropout_rate=best_config.get("dropout_rate", 0.2),  # Default to 0.2 if not specified
            activation_fn=best_config.get("activation_fn", "relu")  # Default to 'relu' if not specified
        )
        # Load model
        model.load_state_dict(torch.load(model_file))
        model.eval()

    # %%
    # List to store decoded 2-gram scores for all test samples

    decoded_test_results_words = []
    result = []
    total_precision = total_recall = total_f1 = total_dice = 0.0
    n = len(dataloader_test.dataset)

    # Switch to evaluation mode (no gradient computation during inference)
    model.eval()

    # Define Threshold for filtering predictions
    threshold = best_config.get("threshold", 0.5)  # Default threshold if not specified

    # Loop through the test dataloader for inference
    with torch.no_grad():  # No need to compute gradients during inference
        for data, labels, uid in tqdm(dataloader_test, desc="Test loop") if GLOBAL_CONFIG["Verbose"] else dataloader_test:

            actual_two_grams = label_tensors_to_two_grams(two_gram_dict, labels)

            # Move data to device and make predictions
            data = data.to(compute_device)
            logits = model(data)
            probabilities = torch.sigmoid(logits)

            # Convert probabilities into 2-gram scores
            batch_two_gram_scores = convert_to_two_gram_scores(two_gram_dict, probabilities)

            # Filter out low-scoring 2-grams
            batch_filtered_two_gram_scores = filter_two_grams(batch_two_gram_scores, threshold)

            # Calculate performance metrics for evaluation
            dice, precision, recall, f1 = calculate_performance_metrics(
                actual_two_grams, batch_filtered_two_gram_scores)
            total_dice += dice
            total_precision += precision
            total_recall += recall
            total_f1 += f1

            for two_grams, two_grams_predicted, uid in zip(actual_two_grams, batch_filtered_two_gram_scores, uid):
                # Create a dictionary to store the results for each test sample
                result_dict = {
                    "uid": uid,
                    "actual_two_grams": two_grams,
                    "filtered_two_grams": two_grams_predicted,
                }
                # Append the result dictionary to the combined results list
                result.append(result_dict)

            average_precision = total_precision / n
            average_recall = total_recall / n
            average_f1 = total_f1 / n
            average_dice = total_dice / n


    # Now `combined_results_performance` contains detailed comparison for all test samples
    print (f"Average Precision: {average_precision}")
    print (f"Average Recall: {average_recall}")
    print (f"Average F1 Score: {average_f1}")
    print (f"Average Dice Similarity: {average_dice}")


    # %%
    # SAVE - Metrics and Result
    if DEA_CONFIG["SaveResults"]:
        with open(metrics_file, "w") as f:
            f.write(f"Average Precision: {average_precision:.4f}\n")
            f.write(f"Average Recall: {average_recall:.4f}\n")
            f.write(f"Average F1 Score: {average_f1:.4f}\n")
            f.write(f"Average Dice Similarity: {average_dice:.4f}\n")

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)


    # %%
    # LOAD - Result
    if DEA_CONFIG["LoadResults"]:
        with open(result_file, 'r', encoding='utf-8') as f:
            result = json.load(f)


    # %% [markdown]
    # ## Visualize Performance for Re-Identification

    # %%
    # Convert results to DataFrame
    results_df = pd.DataFrame(result)

    # Calculate per-sample 2-gram overlap metrics
    overlap_df = pd.DataFrame([two_gram_overlap(row) for _, row in results_df.iterrows()])

    # Plot 1: Distribution of precision, recall, F1
    plt.figure(figsize=(10, 6))
    sns.histplot(overlap_df[['precision', 'recall', 'f1']], bins=20, kde=True, palette='Set2', element="step", fill=False)
    plt.title('Distribution of Precision, Recall, F1 across Samples')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend(['Precision', 'Recall', 'F1'])
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 2: Sample examples of reconstruction
    print("\n🔍 Sample Reconstructions (first 5):")
    for idx, row in results_df.head(5).iterrows():
        print(f"UID: {row['uid']}")
        print(f"  Actual 2-grams:       {row['actual_two_grams']}")
        print(f"  Predicted 2-grams:    {row['filtered_two_grams']}")
        print("-" * 60)

    # %% [markdown]
    # ## Step 6: Refinement and Reconstruction

    # %%
    def run_reidentification(reconstructed, not_reidentified_path, identifier, merge_cols):
        df_not_reidentified = load_not_reidentified_data(not_reidentified_path, identifier)
        df_not_reidentified = lowercase_df(df_not_reidentified)

        df_reconstructed = pd.DataFrame(reconstructed, columns=merge_cols)
        df_reconstructed = lowercase_df(df_reconstructed)

        return reidentification_analysis(df_reconstructed, df_not_reidentified, merge_cols, len(df_not_reidentified), DEA_CONFIG["MatchingTechnique"], save_path=f"{save_to}/re_identification_results")

    match DEA_CONFIG["MatchingTechnique"]:
        case "ai":
            print("\n🔄 Reconstructing results using AI...")
            reconstructed_results = reconstruct_using_ai(result)
            merge_cols = ["GivenName", "Surname", "Birthday", "uid"]

        case "ai_from_reconstructed_strings":
            print("\n🔄 Reconstructing results using AI with preprocessed strings...")
            greedy_result = longest_path_reconstruction(result)
            reconstructed_results = reconstruct_using_ai_from_reconstructed_strings(greedy_result)
            merge_cols = ["GivenName", "Surname", "Birthday", "uid"]

        case "greedy":
            print("\n🔄 Reconstructing results using greedy algorithm...")
            reconstructed_results = longest_path_reconstruction(result)

            df_not_reidentified = load_not_reidentified_data(data_dir, identifier)
            df_not_reidentified['identifier'] = create_identifier_column(df_not_reidentified)
            df_not_reidentified = df_not_reidentified[['uid', 'identifier']]

            df_reconstructed = pd.DataFrame(reconstructed_results).rename(columns={"reconstructed_2grams": "identifier"})
            reidentification_analysis(df_reconstructed, df_not_reidentified, ["uid", "identifier"], len(df_not_reidentified), DEA_CONFIG["MatchingTechnique"], save_path=f"{save_to}/re_identification_results")

            raise SystemExit("✅ Done with greedy mode, exiting early")

        case "fuzzy":
            print("\n🔄 Reconstructing results using fuzzy matching...")
            best_matches_given_name, updated_result = find_most_likely_given_name(result, beginYear=1900, endYear=2024, similarity_metric='dice')
            best_matches_surnames, updated_result = find_most_likely_surnames(predicted_2grams_list=updated_result, minCount=10, similarity_metric='dice', use_filtered_surnames=False)
            best_matches_birthday = find_most_likely_birthday(updated_result, similarity_metric='dice')

            reconstructed_results = [
                (given[0], surname[0], birthday[0], given[2])
                for given, surname, birthday in zip(best_matches_given_name, best_matches_surnames, best_matches_birthday)
            ]
            merge_cols = ["GivenName", "Surname", "Birthday", "uid"]

    # Run final reidentification for non-greedy cases
    if DEA_CONFIG["MatchingTechnique"] != "greedy":
        reidentified = run_reidentification(reconstructed_results, data_dir, identifier, merge_cols)

    # %%
    return






