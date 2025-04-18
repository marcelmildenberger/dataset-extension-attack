def dataset_extension_attack(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG, DEA_CONFIG):
        # %% [markdown]
    # # Privacy-Preserving Record Linkage (PPRL): Investigating Dataset Extension Attacks

    # %% [markdown]
    # ## Imports
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

    import torch
    from torch.utils.tensorboard import SummaryWriter
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split
    import torchvision

    from utils import get_hashes, extract_two_grams, reconstruct_words, precision_recall_f1, dice_coefficient

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
    from datasets.two_step_hash_dataset import TwoStepHashDataset

    from pytorch_models.bloom_filter_to_two_gram_classifier import BloomFilterToTwoGramClassifier
    from pytorch_models.tab_min_hash_to_two_gram_classifier import TabMinHashToTwoGramClassifier
    from pytorch_models.two_step_hash_to_two_gram_classifier import TwoStepHashToTwoGramClassifier
    from pytorch_models.test_model import TestModel

    from early_stopping.early_stopping import EarlyStopping

    print('System Version:', sys.version)
    print('PyTorch version', torch.__version__)
    print('Torchvision version', torchvision.__version__)
    print('Numpy version', np.__version__)
    print('Pandas version', pd.__version__)

    # %% [markdown]
    # ## 🔍 Data Preparation: Load or Compute Graph Matching Attack (GMA) Results
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
    # Get unique hash identifiers for the encoding and embedding configurations
    eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash = get_hashes(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG)

    # Define file paths based on the configuration hashes
    path_reidentified = f"./data/available_to_eve/reidentified_individuals_{eve_enc_hash}_{alice_enc_hash}_{eve_emb_hash}_{alice_emb_hash}.h5"
    path_not_reidentified = f"./data/available_to_eve/not_reidentified_individuals_{eve_enc_hash}_{alice_enc_hash}_{eve_emb_hash}_{alice_emb_hash}.h5"
    path_all = f"./data/dev/alice_data_complete_with_encoding_{eve_enc_hash}_{alice_enc_hash}_{eve_emb_hash}_{alice_emb_hash}.h5"

    # Check if the output files already exist
    if os.path.isfile(path_reidentified) and os.path.isfile(path_not_reidentified) and os.path.isfile(path_all):
        # Load previously saved attack results
        print("Loading previously saved attack results...")
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

    # %% [markdown]
    # ## 🔤 Create 2-Gram Dictionary (Letters & Digits)
    #
    # This code creates a comprehensive dictionary of all possible **2-grams** (two-character combinations) that consist of lowercase letters and digits.
    #
    # 1. **Character Sets:**
    #    - `string.ascii_lowercase`: the lowercase English alphabet ('a' to 'z')
    #    - `string.digits`: the digits '0' to '9'
    #
    # 2. **2-Gram Types Generated:**
    #    - **Letter-Letter (LL):** All combinations like `'aa'`, `'ab'`, ..., `'zz'` (26×26 = 676)
    #    - **Digit-Digit (DD):** All combinations like `'00'`, `'01'`, ..., `'99'` (10×10 = 100)
    #    - **Letter-Digit (LD):** All combinations like `'a0'`, `'a1'`, ..., `'z9'` (26×10 = 260)
    #
    # 3. **Combining All 2-Grams:**
    #    - All three types are concatenated into a single list.
    #
    # 4. **Indexed Dictionary:**
    #    - The `enumerate()` function is used to assign each 2-gram a unique index in `two_gram_dict`.
    #

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
    # ## 🧩 Dataset Creation Based on Alice’s Encoding Scheme
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
    # 1️ Bloom Filter Encoding
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
        bloomfilter_length = len(df_reidentified["bloomfilter"][0])

    # 2️ Tabulation MinHash Encoding
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
        tabminhash_length = len(df_reidentified["tabminhash"][0])

    # 3 Two-Step Hash Encoding (One-Hot Encoding Mode)
    elif ENC_CONFIG["AliceAlgo"] == "TwoStepHash":
        # Collect all unique integers across both reidentified and non-reidentified data
        unique_ints_reid = set().union(*df_reidentified["twostephash"])
        unique_ints_not_reid = set().union(*df_not_reidentified["twostephash"])
        unique_ints_sorted = sorted(unique_ints_reid.union(unique_ints_not_reid))
        unique_integers_dict = {i: val for i, val in enumerate(unique_ints_sorted)}

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

    # %% [markdown]
    # ## Data Splitting & Loader Setup
    #
    # After preprocessing the encoded data, we divide it into training, validation, and test sets using PyTorch's `DataLoader` and `random_split`.
    #
    # ### Dataset Proportions
    # - The proportion for the training set is defined in `DEA_CONFIG["TrainSize"]`.
    # - The remainder is used for validation.
    #
    # ### Splitting
    # - `data_labeled` (the reidentified individuals) is split into:
    #   - `data_train` for training
    #   - `data_val` for validation
    # - `data_not_labeled` (unidentified individuals) is used exclusively for testing.
    #
    # ### Dataloader Configuration
    # - **Training Loader**: shuffled for learning generalization.
    # - **Validation Loader**: also shuffled to vary batches during evaluation.
    # - **Test Loader**: also shuffled.
    #

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
        shuffle=True  # Allows variation in validation batches
    )

    dataloader_test = DataLoader(
        data_not_labeled,
        batch_size=DEA_CONFIG["BatchSize"],
        shuffle=True
    )

    # %% [markdown]
    # ## Model Instantiation Based on Encoding Scheme
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
    # Instantiate model based on selected encoding scheme

    if DEA_CONFIG["TestModel"]:
        if ENC_CONFIG["AliceAlgo"] == "BloomFilter":
            model = TestModel(
                input_dim=bloomfilter_length,
                output_dim=len(all_two_grams),
                hidden_layer=2048,
                num_layers=1,
                dropout_rate=0.220451802221184,
                activation_fn="relu"
            )
        elif ENC_CONFIG["AliceAlgo"] == "TabMinHash":
            model = TestModel(
                input_dim=bloomfilter_length,
                output_dim=len(all_two_grams),
                hidden_layer=2048,
                num_layers=1,
                dropout_rate=0.220451802221184,
                activation_fn="relu"
            )
        elif ENC_CONFIG["AliceAlgo"] == "TwoStepHash":
            model = TestModel(
                input_dim=bloomfilter_length,
                output_dim=len(all_two_grams),
                hidden_layer=2048,
                num_layers=1,
                dropout_rate=0.220451802221184,
                activation_fn="relu"
            )

    elif ENC_CONFIG["AliceAlgo"] == "BloomFilter":
        model = BloomFilterToTwoGramClassifier(
            input_dim=bloomfilter_length,
            output_dim=len(all_two_grams)
        )

    elif ENC_CONFIG["AliceAlgo"] == "TabMinHash":
        model = TabMinHashToTwoGramClassifier(
            input_dim=tabminhash_length,
            output_dim=len(all_two_grams)
        )

    elif ENC_CONFIG["AliceAlgo"] == "TwoStepHash":
        model = TwoStepHashToTwoGramClassifier(
            input_dim=len(unique_ints_sorted),
            output_dim=len(all_two_grams)
        )

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
    # Setup tensorboard logging
    run_name = "".join([
        DEA_CONFIG["LossFunction:"],
        DEA_CONFIG["Optimizer"],
        ENC_CONFIG["AliceAlgo"],
        DEA_CONFIG["ActivationFunction"],
    ])
    tb_writer = SummaryWriter(f"runs/{run_name}")

    # Setup compute device (GPU/CPU)
    compute_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(compute_device)

    # Initialize loss function
    match DEA_CONFIG["LossFunction:"]:
        case "BCEWithLogitsLoss":
            criterion = nn.BCEWithLogitsLoss(reduction='mean')
        case "MultiLabelSoftMarginLoss":
            criterion = nn.MultiLabelSoftMarginLoss(reduction='mean')
        case _:
            raise ValueError(f"Unsupported loss function: {DEA_CONFIG['LossFunction:']}")

    # Initialize optimizer
    match DEA_CONFIG["Optimizer"]:
        case "Adam":
            optimizer = optim.Adam(model.parameters(), lr=DEA_CONFIG["LearningRate"])
        case "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=DEA_CONFIG["LearningRate"])
        case "SGD":
            optimizer = optim.SGD(model.parameters(),
                                lr=DEA_CONFIG["LearningRate"],
                                momentum=DEA_CONFIG["Momentum"])
        case "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=DEA_CONFIG["LearningRate"])
        case _:
            raise ValueError(f"Unsupported optimizer: {DEA_CONFIG['Optimizer']}")

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
        train_losses, val_losses = [], []
        early_stopper = EarlyStopping(patience=DEA_CONFIG["Patience"], min_delta=DEA_CONFIG["MinDelta"])

        for epoch in range(DEA_CONFIG["Epochs"]):
            # Training phase
            model.train()
            train_loss = run_epoch(
                model, dataloader_train, criterion, optimizer,
                device, is_training=True
            )
            train_losses.append(train_loss)

            # Validation phase
            model.eval()
            val_loss = run_epoch(
                model, dataloader_val, criterion, optimizer,
                device, is_training=False
            )
            val_losses.append(val_loss)

            # Logging
            log_metrics(train_loss, val_loss, epoch, DEA_CONFIG["Epochs"])

            # Early stopping check
            if early_stopper(val_loss):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        return train_losses, val_losses

    def run_epoch(model, dataloader, criterion, optimizer, device, is_training):
        running_loss = 0.0
        with torch.set_grad_enabled(is_training):
            for data, labels, _ in tqdm(dataloader,
                                    desc="Training" if is_training else "Validation"):
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

    def log_metrics(train_loss, val_loss, epoch, total_epochs):
        print(f"Epoch {epoch + 1}/{total_epochs} - "
            f"Train loss: {train_loss:.4f}, "
            f"Validation loss: {val_loss:.4f}")
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
    # ## Model Inference and 2-Gram Comparison
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
    # List to store decoded 2-gram scores for all test samples
    decoded_test_results_words = []
    combined_results_performance = []
    total_precision = total_recall = total_f1 = total_dice = 0.0

    # Switch to evaluation mode (no gradient computation during inference)
    model.eval()

    # Define Threshold for filtering predictions
    threshold = DEA_CONFIG["FilterThreshold"]

    # Loop through the test dataloader for inference
    with torch.no_grad():  # No need to compute gradients during inference
        for data_batch, uids in tqdm(dataloader_test, desc="Test loop"):
            # Filter relevant individuals from the dataset based on UIDs
            filtered_df = df_all[df_all["uid"].isin(uids)].drop(df_all.columns[-2], axis=1) # Drop encoding column

            # Extract 2-grams from actual data for comparison
            actual_two_grams_batch = []
            for _, entry in filtered_df.iterrows():
                row = entry[:-1]  # Exclude UID from row
                extracted_two_grams = extract_two_grams("".join(map(str, row)))  # Extract 2-grams from the row
                actual_two_grams_batch.append({"uid": entry["uid"], "two_grams": extracted_two_grams})

            # Move the batch of data to the device (e.g., GPU)
            data_batch = data_batch.to(compute_device)

            # Apply the model to get logits (raw predictions)
            logits = model(data_batch)

            # Convert logits to probabilities using sigmoid (binary classification)
            probabilities = torch.sigmoid(logits)

            # Convert probabilities into 2-gram scores (using the two_gram_dict to map to 2-gram labels)
            batch_two_gram_scores = [
                {two_gram_dict[j]: score.item() for j, score in enumerate(probabilities[i])}  # Map each probability to its 2-gram
                for i in range(probabilities.size(0))  # Iterate over each sample in the batch
            ]

            # Apply threshold to filter out low-scoring 2-grams
            batch_filtered_two_gram_scores = [
                {two_gram: score for two_gram, score in two_gram_scores.items() if score > threshold}  # Only keep scores above threshold
                for two_gram_scores in batch_two_gram_scores
            ]

            # Filtered 2-grams per UID in the batch
            filtered_two_grams = [
                {"uid": uid, "two_grams": {key for key in two_grams.keys()}}  # Only keep the 2-gram keys (no scores)
                for uid, two_grams in zip(uids, batch_filtered_two_gram_scores)
            ]

            # Reconstruct words from the filtered 2-grams for each sample
            batch_reconstructed_words = [
                reconstruct_words(filtered_scores) for filtered_scores in batch_filtered_two_gram_scores
            ]

            # Append the reconstructed words to the results list
            decoded_test_results_words.extend(batch_reconstructed_words)

            # Compare predicted 2-grams with actual 2-grams and calculate performance metrics
            for entry_two_grams_batch in actual_two_grams_batch:  # Loop through each UID in the batch
                for entry_filtered_two_grams in filtered_two_grams:
                    if entry_two_grams_batch["uid"] == entry_filtered_two_grams["uid"]:
                        # Calculate Dice similarity between actual and predicted 2-grams
                        precision, recall, f1 = precision_recall_f1(
                            entry_two_grams_batch["two_grams"],
                            entry_filtered_two_grams["two_grams"]
                        )
                        dice = dice_coefficient(
                            entry_two_grams_batch["two_grams"],
                            entry_filtered_two_grams["two_grams"]
                        )
                        total_precision += precision
                        total_recall += recall
                        total_f1 += f1
                        total_dice += dice

                        combined_results_performance.append({
                            "uid": entry_two_grams_batch["uid"],
                            "actual_two_grams": entry_two_grams_batch["two_grams"],  # Get actual 2-grams for this UID
                            "predicted_two_grams": entry_filtered_two_grams["two_grams"],  # Get predicted 2-grams for this UID
                            "dice_similarity": dice,
                            "precision": precision,
                            "recall": recall,
                            "f1_score": f1,
                        })
            n = len(combined_results_performance)

            average_precision = total_precision / n
            average_recall = total_recall / n
            average_f1 = total_f1 / n
            average_dice = total_dice / n


    # Now `combined_results_performance` contains detailed comparison for all test samples
    print(combined_results_performance)
    print (f"Average Precision: {average_precision}")
    print (f"Average Recall: {average_recall}")
    print (f"Average F1 Score: {average_f1}")
    print (f"Average Dice Similarity: {average_dice}")


    # %%
    sys.exit("Stopping execution at this cell.")

    # %% [markdown]
    # ## Visualize Performance for Re-Identification

    # %% [markdown]
    # ## Testing Area






