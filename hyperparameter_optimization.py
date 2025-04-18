def hyperparameter_optimization(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG, DEA_CONFIG):
        # %% [markdown]
    # # Hyperparameter Optimization

    # %%
    import os

    from datetime import datetime
    import seaborn as sns

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split

    import ray
    from ray import tune
    from ray import train
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.schedulers import ASHAScheduler


    from utils import get_hashes, extract_two_grams, precision_recall_f1, dice_coefficient

    import matplotlib.pyplot as plt # For data viz
    import pandas as pd
    import hickle as hkl
    import string

    from graphMatching.gma import run_gma

    from datasets.bloom_filter_dataset import BloomFilterDataset
    from datasets.tab_min_hash_dataset import TabMinHashDataset
    from datasets.two_step_hash_dataset import TwoStepHashDataset

    from pytorch_models_hyperparameter_optimization.base_model import BaseModel

    # %%
    # Get unique hash identifiers for the encoding and embedding configurations
    eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash = get_hashes(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG)

    # Define file paths based on the configuration hashes
    path_reidentified = f"./data/available_to_eve/reidentified_individuals_{eve_enc_hash}_{alice_enc_hash}_{eve_emb_hash}_{alice_emb_hash}.h5"
    path_not_reidentified = f"./data/available_to_eve/not_reidentified_individuals_{eve_enc_hash}_{alice_enc_hash}_{eve_emb_hash}_{alice_emb_hash}.h5"
    path_all = f"./data/dev/alice_data_complete_with_encoding_{eve_enc_hash}_{alice_enc_hash}_{eve_emb_hash}_{alice_emb_hash}.h5"

    # Check if the output files already exist
    if os.path.isfile(path_reidentified) and os.path.isfile(path_not_reidentified) and os.path.isfile(path_all):
        # Load previously saved attack results
        reidentified_data = hkl.load(path_reidentified)
        not_reidentified_data = hkl.load(path_not_reidentified)
        all_data = hkl.load(path_all)

    else:
        # Run Graph Matching Attack if files are not found
        reidentified_data, not_reidentified_data, all_data = run_gma(
            GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG, DEA_CONFIG,
            eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash
        )

    # Convert lists to DataFrames
    df_reidentified = pd.DataFrame(reidentified_data[1:], columns=reidentified_data[0])
    df_not_reidentified = pd.DataFrame(not_reidentified_data[1:], columns=not_reidentified_data[0])
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
        data_not_labeled = BloomFilterDataset(
            df_not_reidentified,
            is_labeled=False,
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
        data_not_labeled = TabMinHashDataset(
            df_not_reidentified,
            is_labeled=False,
            all_two_grams=all_two_grams,
            dev_mode=GLOBAL_CONFIG["DevMode"]
        )
        input_layer_size = len(df_reidentified["tabminhash"][0])

    # 3 Two-Step Hash Encoding (One-Hot Encoding Mode)
    elif ENC_CONFIG["AliceAlgo"] == "TwoStepHash":
        # Collect all unique integers across both reidentified and non-reidentified data
        unique_ints_reid = set().union(*df_reidentified["twostephash"])
        unique_ints_not_reid = set().union(*df_not_reidentified["twostephash"])
        unique_ints_sorted = sorted(unique_ints_reid.union(unique_ints_not_reid))
        unique_integers_dict = {i: val for i, val in enumerate(unique_ints_sorted)}
        input_layer_size = len(unique_ints_sorted)

        data_labeled = TwoStepHashDataset(
            df_reidentified,
            is_labeled=True,
            all_integers=unique_ints_sorted,
            all_two_grams=all_two_grams,
            dev_mode=GLOBAL_CONFIG["DevMode"]
        )
        data_not_labeled = TwoStepHashDataset(
            df_not_reidentified,
            is_labeled=False,
            all_integers=unique_ints_sorted,
            all_two_grams=all_two_grams,
            dev_mode=GLOBAL_CONFIG["DevMode"]
        )

    # %%
    # Define dataset split proportions
    train_size = int(DEA_CONFIG["TrainSize"] * len(data_labeled))
    val_size = len(data_labeled) - train_size

    # Split the reidentified dataset into training and validation sets
    data_train, data_val = random_split(data_labeled, [train_size, val_size])

    # Create DataLoaders for training, validation, and testing
    dataloader_train = DataLoader(
        data_train,
        batch_size=DEA_CONFIG["BatchSize"],
        shuffle=True  # Important for training
    )

    dataloader_val = DataLoader(
        data_val,
        batch_size=DEA_CONFIG["BatchSize"],
        shuffle=False  # Allows variation in validation batches
    )

    dataloader_test = DataLoader(
        data_not_labeled,
        batch_size=DEA_CONFIG["BatchSize"],
        shuffle=False
    )

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
    def train_model(config, dataloader_tr, dataloader_v, dataloader_te, threshold):
        # Initialize lists to store training and validation losses
        train_losses, val_losses = [], []
        total_precision = total_recall = total_f1 = total_dice = 0.0
        n = 0

        # Define and initialize model with hyperparameters from config
        model = BaseModel(
            input_dim=input_layer_size,
            output_dim=len(all_two_grams),
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
            "MultiLabelSoftMarginLoss": nn.MultiLabelSoftMarginLoss()
        }
        criterion = loss_functions[config["loss_fn"]]

        # Select optimizer based on config
        optimizers = {
            "Adam": optim.Adam(model.parameters(), lr=config["lr"]),
            "SGD": optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9),
            "RMSprop": optim.RMSprop(model.parameters(), lr=config["lr"])
        }
        optimizer = optimizers[config["optimizer"]]

        # Training loop
        for epoch in range(config["epochs"]):
            # Training phase
            model.train()
            train_loss = run_epoch(model, dataloader_tr, criterion, optimizer, device, is_training=True)
            train_losses.append(train_loss)

            # Validation phase
            model.eval()
            val_loss = run_epoch(model, dataloader_v, criterion, optimizer, device, is_training=False)
            val_losses.append(val_loss)

        # Test phase with reconstruction and evaluation
        model.eval()

        with torch.no_grad():
            for data_batch, uids in dataloader_te:
                # Process the test data
                filtered_df = df_all[df_all["uid"].isin(uids)].drop(df_all.columns[-2], axis=1)
                actual_two_grams_batch = extract_two_grams_batch(filtered_df)

                # Move data to device and make predictions
                data_batch = data_batch.to(device)
                logits = model(data_batch)
                probabilities = torch.sigmoid(logits)

                # Convert probabilities into 2-gram scores
                batch_two_gram_scores = convert_to_two_gram_scores(probabilities)

                # Filter out low-scoring 2-grams
                batch_filtered_two_gram_scores = filter_two_grams(batch_two_gram_scores, threshold)
                filtered_two_grams_with_uid = combine_two_grams_with_uid(uids, batch_filtered_two_gram_scores)


                dice, precision, recall, f1 = calculate_performance_metrics(
                    actual_two_grams_batch, filtered_two_grams_with_uid)
                # Calculate Dice similarity for evaluation
                total_dice += dice
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                n += data_batch.size(0)

        train.report({
                "precision": total_precision,
                "recall": total_recall,
                "f1": total_f1,
                "dice": total_dice,
                "average_dice": total_dice / n,
                "average_precision": total_precision / n,
                "average_recall": total_recall / n,
                "average_f1": total_f1 / n,
                "len_train": len(dataloader_tr.dataset),
                "len_val": len(dataloader_v.dataset),
                "len_test": len(dataloader_te.dataset),
            })



    def run_epoch(model, dataloader, criterion, optimizer, device, is_training):
        running_loss = 0.0
        with torch.set_grad_enabled(is_training):
            for data, labels, _ in dataloader:
                data, labels = data.to(device), labels.to(device)
                if is_training:
                    optimizer.zero_grad()

                outputs = model(data)
                loss = criterion(outputs, labels)

                if is_training:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * labels.size(0)

        return running_loss / len(dataloader.dataset)


    def extract_two_grams_batch(df):
        return [
            {"uid": entry["uid"], "two_grams": extract_two_grams("".join(map(str, entry[:-1])))}
            for _, entry in df.iterrows()
        ]


    def convert_to_two_gram_scores(probabilities):
        return [
            {two_gram_dict[j]: score.item() for j, score in enumerate(probabilities[i])}
            for i in range(probabilities.size(0))
        ]


    def filter_two_grams(two_gram_scores, threshold):
        return [
            {two_gram: score for two_gram, score in two_gram_scores.items() if score > threshold}
            for two_gram_scores in two_gram_scores
        ]


    def combine_two_grams_with_uid(uids, filtered_two_gram_scores):
        return [
            {"uid": uid, "two_grams": {key for key in two_grams.keys()}}
            for uid, two_grams in zip(uids, filtered_two_gram_scores)
        ]

    def calculate_performance_metrics(actual_two_grams_batch, filtered_two_grams):
        sum_precision = 0
        sum_recall = 0
        sum_f1 = 0
        sum_dice = 0
        for entry_two_grams_batch in actual_two_grams_batch:
            for entry_filtered_two_grams in filtered_two_grams:
                if entry_two_grams_batch["uid"] == entry_filtered_two_grams["uid"]:
                    precision, recall, f1 = precision_recall_f1(entry_two_grams_batch["two_grams"], entry_filtered_two_grams["two_grams"])
                    sum_dice += dice_coefficient(entry_two_grams_batch["two_grams"], entry_filtered_two_grams["two_grams"])
                    sum_precision += precision
                    sum_recall += recall
                    sum_f1 += f1
        return sum_dice, sum_precision, sum_recall, sum_f1


    # %%
    # Define search space for hyperparameter optimization
    search_space = {
        "num_layers": tune.randint(1, 8),  # Vary the number of layers in the model
        "hidden_layer_size": tune.choice([128, 256, 512, 1024, 2048]),  # Different sizes for hidden layers
        "dropout_rate": tune.uniform(0.1, 0.4),  # Dropout rate between 0.1 and 0.4
        "activation_fn": tune.choice(["relu", "leaky_relu", "gelu"]),  # Activation functions to choose from
        "optimizer": tune.choice(["Adam", "SGD", "RMSprop"]),  # Optimizer options
        "loss_fn": tune.choice(["BCEWithLogitsLoss"]),  # Loss function (currently using BCEWithLogitsLoss)
        "lr": tune.loguniform(1e-5, 1e-2),  # Learning rate in a log-uniform range
        "epochs": tune.randint(5, 30),  # Number of epochs between 10 and 20
    }

    # Initialize Ray for hyperparameter optimization
    ray.init(ignore_reinit_error=True)

    # Optuna Search Algorithm for optimizing the hyperparameters
    optuna_search = OptunaSearch(metric="dice", mode="max")

    # Use ASHAScheduler to manage trials and early stopping
    scheduler = ASHAScheduler(metric="dice", mode="max")

    trainable = tune.with_parameters(
        train_model,
        dataloader_tr = dataloader_train,
        dataloader_v = dataloader_val,
        dataloader_te = dataloader_test,
        threshold = DEA_CONFIG["FilterThreshold"]
    )


    # Define and configure the Tuner for Ray Tune
    tuner = tune.Tuner(
        trainable,  # The function to optimize (training function)
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,  # Search strategy using Optuna
            scheduler=scheduler,  # Use ASHA to manage the trials
            num_samples=DEA_CONFIG["NumSamples"],  # Number of trials to run
            max_concurrent_trials=2  # or 1 if your models are heavy
        ),
        param_space=search_space  # Pass in the defined hyperparameter search space
    )

    # Run the tuner
    results = tuner.fit()

    # Shut down Ray after finishing the optimization
    ray.shutdown()

    # %%
    selected_dataset = GLOBAL_CONFIG["Data"].split("/")[-1].replace(".tsv", "")


    experiment_tag = "experiment_" + ENC_CONFIG["AliceAlgo"] + "_" + selected_dataset + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_to = f"experiment_results/{experiment_tag}"
    os.makedirs(save_to, exist_ok=True)
    result_grid = results

    # Best and worst result based on dice
    best_result = result_grid.get_best_result(metric="dice", mode="max")
    worst_result = result_grid.get_best_result(metric="dice", mode="min")

    # Combine configs and metrics into a DataFrame
    df = pd.DataFrame([
        {
            **result.config,
            **{k: result.metrics.get(k) for k in ["precision", "recall", "f1", "dice", "average_dice", "average_precision", "average_recall", "average_f1"]},
        }
        for result in result_grid
    ])

    # Save to CSV
    df.to_csv(f"{save_to}/all_trial_results.csv", index=False)
    print("✅ Results saved to all_trial_results.csv")

    def print_result(label, result):
        print(f"\n🔍 {label}")
        print("-" * 40)
        print(f"Config: {result.config}")
        print(f"Dice Score: {result.metrics.get('dice'):.4f}")
        print(f"Precision:  {result.metrics.get('precision'):.4f}")
        print(f"Recall:     {result.metrics.get('recall'):.4f}")
        print(f"F1 Score:   {result.metrics.get('f1'):.4f}")
        print(f"Average Dice: {result.metrics.get('average_dice'):.4f}")
        print(f"Average Precision: {result.metrics.get('average_precision'):.4f}")
        print(f"Average Recall: {result.metrics.get('average_recall'):.4f}")
        print(f"Average F1: {result.metrics.get('average_f1'):.4f}")

        result_dict = {**result.config, **result.metrics}
        result_dict.pop("config", None)
        result_dict.pop("checkpoint_dir_name", None)
        result_dict.pop("experiment_tag", None)
        # Convert to a DataFrame and save
        df = pd.DataFrame([result_dict])
        df.to_csv(f"{save_to}/{label}.csv", index=False)

    print_result("Best_Result", best_result)
    print_result("Worst_Result", worst_result)

    # Compute and print average metrics
    print("\n📊 Average Metrics Across All Trials")
    avg_metrics = df[["precision", "recall", "f1", "dice", "average_dice", "average_precision", "average_recall", "average_f1"]].mean()
    print("-" * 40)
    for key, value in avg_metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")

    # --- 📈 Plotting performance metrics ---
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[["precision", "recall", "f1", "dice"]])
    plt.title("Distribution of Performance Metrics Across Trials")
    plt.grid(True)
    plt.savefig(f"{save_to}/metric_distributions.png")
    print("📊 Saved plot: metric_distributions.png")

    # --- 📌 Correlation between config params and performance ---
    # Only include numeric config columns
    numeric_config_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    correlation_df = df[numeric_config_cols].corr()

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_df, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Between Parameters and Metrics")
    plt.tight_layout()
    plt.savefig(f"{save_to}/correlation_heatmap.png")
    print("📌 Saved heatmap: correlation_heatmap.png")

    return None



