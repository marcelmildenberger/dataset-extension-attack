def dataset_extension_attack(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG, DEA_CONFIG):
    # %% [markdown]
    # # Hyperparameter Optimization

    # %%
    import os

    from datetime import datetime
    import seaborn as sns

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split, Subset

    import ray
    from ray import tune, air
    from ray import train
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.schedulers import ASHAScheduler




    from utils import get_hashes, convert_to_two_gram_scores, filter_two_grams, calculate_performance_metrics, run_epoch, label_tensors_to_two_grams

    import matplotlib.pyplot as plt # For data viz
    import pandas as pd
    import hickle as hkl
    import string
    from early_stopping.early_stopping import EarlyStopping

    from graphMatching.gma import run_gma

    from datasets.bloom_filter_dataset import BloomFilterDataset
    from datasets.tab_min_hash_dataset import TabMinHashDataset
    from datasets.two_step_hash_dataset import TwoStepHashDataset

    from pytorch_models_hyperparameter_optimization.base_model import BaseModel

    print('PyTorch version', torch.__version__)

    # %%
    # Get unique hash identifiers for the encoding and embedding configurations
    eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash = get_hashes(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG)

    # Define file paths based on the configuration hashes
    path_reidentified = f"./data/available_to_eve/reidentified_individuals_{eve_enc_hash}_{alice_enc_hash}_{eve_emb_hash}_{alice_emb_hash}.h5"
    path_all = f"./data/dev/alice_data_complete_with_encoding_{eve_enc_hash}_{alice_enc_hash}_{eve_emb_hash}_{alice_emb_hash}.h5"

    # Check if the output files already exist
    if os.path.isfile(path_reidentified) and os.path.isfile(path_all):
        # Load previously saved attack results
        reidentified_data = hkl.load(path_reidentified)
        all_data = hkl.load(path_all)

    else:
        # Run Graph Matching Attack if files are not found
        reidentified_data, _, all_data = run_gma(
            GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG, DEA_CONFIG,
            eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash
        )

    # Convert lists to DataFrames
    df_reidentified = pd.DataFrame(reidentified_data[1:], columns=reidentified_data[0])
    df_all = pd.DataFrame(all_data[1:], columns=all_data[0])

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

    # %%
    # 1️⃣ Bloom Filter Encoding
    if ENC_CONFIG["AliceAlgo"] == "BloomFilter":
        data_labeled = BloomFilterDataset(
            df_reidentified,
            is_labeled=True,
            all_two_grams=all_two_grams,
            dev_mode=GLOBAL_CONFIG["DevMode"]
        )
        input_layer_size = len(df_reidentified["bloomfilter"][0])

    # 2️⃣ Tabulation MinHash Encoding
    elif ENC_CONFIG["AliceAlgo"] == "TabMinHash":
        data_labeled = TabMinHashDataset(
            df_reidentified,
            is_labeled=True,
            all_two_grams=all_two_grams,
            dev_mode=GLOBAL_CONFIG["DevMode"]
        )
        input_layer_size = len(df_reidentified["tabminhash"][0])

    # 3 Two-Step Hash Encoding (One-Hot Encoding Mode)
    elif ENC_CONFIG["AliceAlgo"] == "TwoStepHash":
        # Collect all unique integers across both reidentified and non-reidentified data
        unique_ints = sorted(set().union(*df_all["twostephash"]))
        unique_integers_dict = {i: val for i, val in enumerate(unique_ints)}
        input_layer_size = len(unique_ints)

        data_labeled = TwoStepHashDataset(
            df_reidentified,
            is_labeled=True,
            all_integers=unique_ints,
            all_two_grams=all_two_grams,
            dev_mode=GLOBAL_CONFIG["DevMode"]
        )

    # %%
    # Define dataset split proportions
    train_size = int(DEA_CONFIG["TrainSize"] * len(data_labeled))
    val_size = len(data_labeled) - train_size

    # Split the reidentified dataset into training and validation sets
    data_train, data_val = random_split(data_labeled, [train_size, val_size])

    # %% [markdown]
    # ## Hyperparameter Tuning Setup for Training
    #
    # This setup for hyperparameter tuning in a neural network model improves modularity, ensuring easy customization for experimentation.
    #
    # 1. **Model Initialization**:
    #    - The model is initialized using hyperparameters from the `config` dictionary, including the number of layers, hidden layer size, dropout rate, and activation function.
    #
    # 2. **Loss Function and Optimizer Selection**:
    #    - The loss function (`criterion`) and optimizer are selected dynamically from the `config` dictionary.
    #
    # 3. **Training & Validation Loop**:
    #    - The training and validation phases are handled in separate loops. The loss is computed at each step, and metrics are logged.
    #
    # 4. **Model Evaluation**:
    #    - After training, the model is evaluated on a test set, where 2-gram predictions are compared against the actual 2-grams.
    #    - **Dice similarity coefficient** is used as a metric to evaluate model performance.
    #
    # 5. **Custom Helper Functions**:
    #    - `extract_two_grams_batch()`: Extracts 2-grams for all samples in the batch.
    #    - `convert_to_two_gram_scores()`: Converts model output logits into 2-gram scores.
    #    - `filter_two_grams()`: Applies a threshold to filter 2-gram scores.
    #    - `filter_two_grams_per_uid()`: Filters and formats 2-gram predictions for each UID.
    #
    # 6. **Hyperparameter Tuning**:
    #    - The setup is integrated with **Ray Tune** (`tune.report`) to enable hyperparameter tuning by reporting the Dice similarity metric.
    #

    # %%
    def train_model(config):
        # Create DataLoaders for training, validation, and testing
        dataloader_train = DataLoader(
            config["data_train"],
            batch_size=DEA_CONFIG["BatchSize"],
            shuffle=True  # Important for training
        )

        dataloader_val = DataLoader(
            config["data_val"],
            batch_size=DEA_CONFIG["BatchSize"],
            shuffle=False  # Allows variation in validation batches
        )

        train_losses = []
        val_losses = []
        total_precision = total_recall = total_f1 = total_dice = 0.0
        n = len(dataloader_val.dataset)
        epochs = 0
        early_stopper = EarlyStopping(patience=DEA_CONFIG["Patience"], min_delta=DEA_CONFIG["MinDelta"])

        # Define and initialize model with hyperparameters from config
        model = BaseModel(
            input_dim=config["input_dim"],
            output_dim=config["output_dim"],
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
            "FocalLoss": nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0)),
            "WeightedBCE": nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0)),
            "CrossEntropyLoss": nn.CrossEntropyLoss(),
            "MSELoss": nn.MSELoss(),
            "L1Loss": nn.L1Loss(),
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
                "len_train": len(dataloader_train.dataset),
                "len_val": len(dataloader_val.dataset),
                "epochs": epochs
            })

    # %%
    # Define search space for hyperparameter optimization
    search_space = {
        "num_layers": tune.randint(1, 8),  # Vary the number of layers in the model
        "hidden_layer_size": tune.choice([64, 128, 256, 512, 1024, 2048]),  # Different sizes for hidden layers
        "dropout_rate": tune.uniform(0.1, 0.4),  # Dropout rate between 0.1 and 0.4
        "activation_fn": tune.choice(["relu", "leaky_relu", "gelu", "elu", "selu", "tanh"]),  # Activation functions to choose from
        "optimizer": tune.choice([
            {"name": "Adam", "lr": tune.loguniform(1e-5, 1e-3)},
            {"name": "AdamW", "lr": tune.loguniform(1e-5, 1e-3)},
            {"name": "SGD", "lr": tune.loguniform(1e-4, 1e-2), "momentum": tune.uniform(0.0, 0.99)},
            {"name": "RMSprop", "lr": tune.loguniform(1e-5, 1e-3)},
        ]),
        "loss_fn": tune.choice(["BCEWithLogitsLoss", "MultiLabelSoftMarginLoss", "FocalLoss", "WeightedBCE"]),
        "threshold": tune.uniform(0.3, 0.8),  # oder: DEA_CONFIG["FilterThreshold"]
        "input_dim": input_layer_size,
        "output_dim": len(all_two_grams),
        "data_train": data_train,
        "data_val": data_val,
        "lr_scheduler": tune.choice([
            {"name": "StepLR", "step_size": tune.choice([5, 10, 20]), "gamma": tune.uniform(0.1, 0.9)},
            {"name": "ExponentialLR", "gamma": tune.uniform(0.85, 0.99)},
            {"name": "ReduceLROnPlateau", "mode": "min", "factor": tune.uniform(0.1, 0.5), "patience": tune.choice([5, 10, 15])},
            {"name": "CosineAnnealingLR", "T_max": tune.loguniform(10, 50) , "eta_min": tune.choice([1e-5, 1e-6, 0])},
            {"name": "CyclicLR", "base_lr": tune.loguniform(1e-5, 1e-3), "max_lr": tune.loguniform(1e-3, 1e-1), "step_size_up": tune.choice([2000, 4000]), "mode_cyclic": tune.choice(["triangular", "triangular2", "exp_range"]) },
            {"name": "None"}  # No scheduler
        ])
    }

    selected_dataset = GLOBAL_CONFIG["Data"].split("/")[-1].replace(".tsv", "")

    experiment_tag = "experiment_" + ENC_CONFIG["AliceAlgo"] + "_" + selected_dataset + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    local_dir = os.path.abspath("./ray_results")

    # Initialize Ray for hyperparameter optimization
    ray.init(ignore_reinit_error=True)

    # Optuna Search Algorithm for optimizing the hyperparameters
    optuna_search = OptunaSearch(metric="average_dice", mode="max")

    # Use ASHAScheduler to manage trials and early stopping
    scheduler = ASHAScheduler(metric="average_dice", mode="max")

    # Define and configure the Tuner for Ray Tune
    tuner = tune.Tuner(
        train_model,  # The function to optimize (training function)
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,  # Search strategy using Optuna
            scheduler=scheduler,  # Use ASHA to manage the trials
            num_samples=DEA_CONFIG["NumSamples"],  # Number of trials to run
            max_concurrent_trials=5  # or 1 if your models are heavy
        ),
        run_config=air.RunConfig(
            name=experiment_tag,
            local_dir=local_dir
        ),
        param_space=search_space  # Pass in the defined hyperparameter search space
    )

    # Run the tuner
    results = tuner.fit()

    # Shut down Ray after finishing the optimization
    ray.shutdown()

    # %%
    # List of keys to remove
    keys_to_remove = [
        "config", "checkpoint_dir_name", "experiment_tag", "done", "training_iteration",
        "trial_id", "date", "time_this_iter_s", "pid", "time_total_s", "hostname",
        "node_ip", "time_since_restore", "iterations_since_restore", "timestamp",
        "input_dim", "output_dim", "data_train", "data_val"
    ]

    def clean_result_dict(result_dict):
        for key in keys_to_remove:
            result_dict.pop(key, None)
        return result_dict

    def resolve_config(config):
        resolved = {}
        for k, v in config.items():
            # If the value is a dictionary, recurse and apply resolve_config
            if isinstance(v, dict):
                resolved[k] = resolve_config(v)
            # If the value is a Ray search sample object (e.g., Float, Categorical)
            elif not isinstance(v, (int, float, str, Subset)):
                resolved[k] = v.sample()  # Get the concrete value from the sample
            else:
                resolved[k] = v  # Leave it as-is if it's not a sample object or Subset
        return resolved

    # %%
    save_to = f"experiment_results/{experiment_tag}"
    os.makedirs(save_to, exist_ok=True)
    result_grid = results

    # Best and worst result based on dice
    best_result = result_grid.get_best_result(metric="average_dice", mode="max")
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
    df.to_csv(f"{save_to}/all_trial_results.csv", index=False)
    print("✅ Results saved to all_trial_results.csv")


    def print_result(label, result):
        print(f"\n🔍 {label}")
        print("-" * 40)
        cleaned_config = resolve_config(result.config)
        print(f"Config: {cleaned_config}")
        print(f"Average Dice: {result.metrics.get('average_dice'):.4f}")
        print(f"Average Precision: {result.metrics.get('average_precision'):.4f}")
        print(f"Average Recall: {result.metrics.get('average_recall'):.4f}")
        print(f"Average F1: {result.metrics.get('average_f1'):.4f}")
        result_dict = {**cleaned_config, **result.metrics}
        clean_result_dict(result_dict)
        # Convert to a DataFrame and save
        df = pd.DataFrame([result_dict])
        df.to_csv(f"{save_to}/{label}.csv", index=False)

    print_result("Best_Result", best_result)
    print_result("Worst_Result", worst_result)

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
    plt.savefig(f"{save_to}/metric_distributions.png")
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
    plt.savefig(f"{save_to}/correlation_heatmap.png")
    print("📌 Saved heatmap: correlation_heatmap.png")



