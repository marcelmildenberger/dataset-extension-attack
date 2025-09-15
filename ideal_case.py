from math import floor
import sys
import copy
import json
import os
import string
import time
from datetime import datetime
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from functools import lru_cache
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import ray
import warnings
from ray import air, train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from early_stopping.early_stopping import EarlyStopping
from graphMatching.gma import run_gma
from datasets.bloom_filter_dataset import BloomFilterDataset
from datasets.tab_min_hash_dataset import TabMinHashDataset
from datasets.two_step_hash_dataset import TwoStepHashDataset
from pytorch_models.base_model import BaseModel
from pytorch_models_hyperparameter_optimization.base_model_hyperparameter_optimization import BaseModelHyperparameterOptimization
from optimal_model_config import initialize_optimal_training_setup, print_optimal_config
from utils import (
    calculate_performance_metrics,
    decode_labels_to_two_grams,
    filter_high_scoring_two_grams,
    get_hashes,
    map_probabilities_to_two_grams,
    metrics_per_entry,
    print_and_save_result,
    read_header,
    resolve_config,
    run_epoch,
    save_dea_runtime_log,
    load_experiment_datasets,
    log_epoch_metrics,
    plot_loss_curves,
    plot_metric_distributions,
    get_not_reidentified_df,
    get_reidentification_techniques,
    run_selected_reidentification,
)

def run_dea(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG, DEA_CONFIG):
    """
    Main experiment entry point. Handles configuration, dataset loading, training, evaluation, and analysis.
    Uses load_experiment_datasets to load train/val/test splits as needed.
    """
    # Set default values for alignment and global configuration.
    # ALIGN_CONFIG["RegWS"] is set to the maximum of 0.1 and one third of the overlap parameter.
    # GLOBAL_CONFIG["Workers"] is set to the number of available CPU cores minus one.
    ALIGN_CONFIG["RegWS"] = max(0.1, GLOBAL_CONFIG["Overlap"] / 3)
    GLOBAL_CONFIG["Workers"] = os.cpu_count()

    # Ignore optuna warnings.
    warnings.filterwarnings("ignore", category=UserWarning, module="optuna")

    # Create a unique experiment directory for saving results and configuration.
    # The directory name encodes the algorithm, dataset, and timestamp for traceability.
    # All configuration dictionaries are saved to a config.txt file in this directory for reproducibility.
    # Save configs as CSV for better analysis
    selected_dataset = GLOBAL_CONFIG["Data"].split("/")[-1].replace(".tsv", "")
    experiment_tag = "experiment_" + ENC_CONFIG["AliceAlgo"] + "_" + selected_dataset + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_experiment_directory = f"experiment_results/{experiment_tag}"
    os.makedirs(current_experiment_directory, exist_ok=True)

    all_configs = {
        "GLOBAL_CONFIG": GLOBAL_CONFIG,
        "DEA_CONFIG": DEA_CONFIG,
        "ENC_CONFIG": ENC_CONFIG,
        "EMB_CONFIG": EMB_CONFIG,
        "ALIGN_CONFIG": ALIGN_CONFIG
    }
    
    # Also save as JSON for reference
    with open(os.path.join(current_experiment_directory, "config.json"), "w") as f:
        json.dump(all_configs, f, indent=4)

    # Generate all possible two-character combinations (2-grams) from lowercase letters and digits.
    # This includes letter-letter, letter-digit, and digit-digit pairs.
    # The resulting list `all_two_grams` is used for encoding/decoding tasks,
    # and `two_gram_dict` maps each index to its corresponding 2-gram.
    alphabet = string.ascii_lowercase
    digits = string.digits
    letter_letter_grams = [a + b for a in alphabet for b in alphabet]
    digit_digit_grams = [d1 + d2 for d1 in digits for d2 in digits]
    letter_digit_grams = [l + d for l in alphabet for d in digits]
    all_two_grams = letter_letter_grams + letter_digit_grams + digit_digit_grams
    two_gram_dict = {i: two_gram for i, two_gram in enumerate(all_two_grams)}

    # Start timing the total run and the GMA run.
    if GLOBAL_CONFIG["BenchMode"]:
        start_total = time.time()

    # Get the hashes for the encoding and embedding.
    eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash = get_hashes(
        GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG
    )

    identifier = f"{eve_enc_hash}_{alice_enc_hash}_{eve_emb_hash}_{alice_emb_hash}"
    data_dir = os.path.abspath("./data")
    dataset_name = GLOBAL_CONFIG["Data"].split("/")[-1].replace(".tsv", "")

    datasets = load_experiment_datasets(dataset_name, DEA_CONFIG["Overlap"], all_two_grams, ENC_CONFIG, GLOBAL_CONFIG, DEA_CONFIG, splits=("train", "val", "test"))

    # Start timing the model training.
    if GLOBAL_CONFIG["BenchMode"]:
        start_model_training = time.time()

    datasets = load_experiment_datasets(dataset_name, DEA_CONFIG["Overlap"], all_two_grams, ENC_CONFIG, GLOBAL_CONFIG, DEA_CONFIG, splits=("train", "val", "test"))
    data_train, data_val, data_test = datasets["train"], datasets["val"], datasets["test"]

    input_dim=data_train[0][0].shape[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize optimal training setup based on encoding scheme
    encoding_scheme = ENC_CONFIG["AliceAlgo"]
    print(f"Initializing optimal configuration for {encoding_scheme}...")
    print_optimal_config(encoding_scheme)
    training_setup = initialize_optimal_training_setup(
        encoding_scheme=encoding_scheme,
        input_dim=input_dim,
        output_dim=len(all_two_grams),
        device=device
    )
    
    model = training_setup["model"]
    optimizer = training_setup["optimizer"]
    criterion = training_setup["criterion"]
    scheduler = training_setup["scheduler"]
    threshold = training_setup["threshold"]
    optimal_batch_size = training_setup["batch_size"]
    
    # Create dataloaders with optimal batch size
    dataloader_train = DataLoader(
        data_train,
        batch_size=optimal_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=GLOBAL_CONFIG["Workers"] // 10 if GLOBAL_CONFIG["UseGPU"] else 0,
    )
    dataloader_val = DataLoader(
        data_val,
        batch_size=optimal_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=GLOBAL_CONFIG["Workers"] // 10 if GLOBAL_CONFIG["UseGPU"] else 0,
    )
    dataloader_test = DataLoader(
        data_test,
        batch_size=optimal_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=GLOBAL_CONFIG["Workers"] // 10 if GLOBAL_CONFIG["UseGPU"] else 0,
    )


    def train_final_model(model, dataloader_train, dataloader_val, criterion, optimizer, device, scheduler=None):
        num_epochs = DEA_CONFIG["Epochs"]
        verbose = GLOBAL_CONFIG["Verbose"]
        patience = DEA_CONFIG["Patience"]
        min_delta = DEA_CONFIG["MinDelta"]
        best_val_loss = float('inf')
        best_model_state = None
        early_stopper = EarlyStopping(patience=patience, min_delta=min_delta, verbose=verbose)
        train_losses, val_losses = [], []

        for epoch in range(num_epochs):
            train_loss = run_epoch(
                model, dataloader_train, criterion, optimizer,
                device, is_training=True, verbose=verbose, scheduler=scheduler
            )
            train_losses.append(train_loss)
            val_loss = run_epoch(
                model, dataloader_val, criterion, optimizer,
                device, is_training=False, verbose=verbose, scheduler=scheduler
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
            val_losses.append(val_loss)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif scheduler is not None:
                scheduler.step()

            log_epoch_metrics(epoch, num_epochs, train_loss, val_loss, tb_writer=None, save_results=GLOBAL_CONFIG["SaveResults"])
            if early_stopper(val_loss):
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                break
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        return model, train_losses, val_losses

    # Train the final model.
    model, train_losses, val_losses = train_final_model(
        model, dataloader_train, dataloader_val,
        criterion, optimizer, device,
        scheduler=scheduler
    )

    # Stop timing the model training.
    if GLOBAL_CONFIG["BenchMode"]:
        elapsed_model_training = time.time() - start_model_training

    # Define the paths for the model, config, result, and metrics files.
    trained_model_directory = os.path.join(
        current_experiment_directory,
        "trained_model"
    )
    os.makedirs(trained_model_directory, exist_ok=True)

    # Plot the training and validation loss curves.
    plot_loss_curves(
        train_losses,
        val_losses,
        save_path=f"{trained_model_directory}/loss_curve.png",
        save=GLOBAL_CONFIG["SaveResults"]
    )

    # Start timing the application to encoded data.
    if GLOBAL_CONFIG["BenchMode"]:
        start_application_to_encoded_data = time.time()

    # Initialize the metrics.
    total_dice = total_precision = total_recall = total_f1 = 0.0
    num_samples = 0

    # Initialize the results.
    results = []

    model.eval()
    dataloader_iter = tqdm(dataloader_test, desc="Test loop") if GLOBAL_CONFIG["Verbose"] else dataloader_test
    with torch.no_grad():
        for data, labels, uids in dataloader_iter:
            data, labels = data.to(device), labels.to(device)
            logits = model(data)
            probs = torch.sigmoid(logits)
            actual_two_grams = decode_labels_to_two_grams(two_gram_dict, labels)
            predicted_scores = map_probabilities_to_two_grams(two_gram_dict, probs)
            predicted_filtered = filter_high_scoring_two_grams(predicted_scores, threshold)
            bs = data.size(0)
            dice, precision, recall, f1 = calculate_performance_metrics(actual_two_grams, predicted_filtered)
            total_dice += dice
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            num_samples += bs
            for uid, actual, predicted in zip(uids, actual_two_grams, predicted_filtered):
                metrics = metrics_per_entry(actual, predicted)
                results.append({
                    "uid": uid,
                    "actual_two_grams": actual,
                    "predicted_two_grams": predicted,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "dice": metrics["dice"],
                    "jaccard": metrics["jaccard"]
                })
    avg_dice = total_dice / num_samples
    avg_precision = total_precision / num_samples
    avg_recall = total_recall / num_samples
    avg_f1 = total_f1 / num_samples

    # Stop timing the application to encoded data.
    if GLOBAL_CONFIG["BenchMode"]:
        elapsed_application_to_encoded_data = time.time() - start_application_to_encoded_data

    # Save the metrics and results if requested.
    if GLOBAL_CONFIG["SaveResults"]:
        # Save metrics as CSV for better analysis
        metrics_data = {
            "metric": ["avg_precision", "avg_recall", "avg_f1", "avg_dice"],
            "value": [avg_precision, avg_recall, avg_f1, avg_dice]
        }
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(f"{trained_model_directory}/metrics.csv", index=False)

    if GLOBAL_CONFIG["SavePredictions"]:
        with open(f"{trained_model_directory}/results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    # Normalize the results.
    results_df = pd.json_normalize(results)

    # Plot the metric distributions.
    plot_metric_distributions(
        results_df,
        trained_model_directory,
        save=GLOBAL_CONFIG["SaveResults"]
    )

    # Start timing the refinement and reconstruction.
    if GLOBAL_CONFIG["BenchMode"]:
        start_refinement_and_reconstruction = time.time()


    header = read_header(GLOBAL_CONFIG["Data"])
    include_birthday = not (GLOBAL_CONFIG["Data"] == "./data/datasets/titanic_full.tsv")
    TECHNIQUES = get_reidentification_techniques(header, include_birthday)
    selected = DEA_CONFIG["MatchingTechnique"]
    df_not_reid_cached = get_not_reidentified_df(dataset_name, DEA_CONFIG["Overlap"])
    re_identification_results_directory = f"{current_experiment_directory}/re_identification_results"

    run_selected_reidentification(
        selected,
        TECHNIQUES,
        results,
        df_not_reid_cached,
        GLOBAL_CONFIG,
        current_experiment_directory,
        data_dir,
        identifier,
        re_identification_results_directory
    )

    # Stop timing the refinement and reconstruction.
    if GLOBAL_CONFIG["BenchMode"]:
        elapsed_refinement_and_reconstruction = time.time() - start_refinement_and_reconstruction
        elapsed_total = time.time() - start_total
        save_dea_runtime_log(
            elapsed_model_training=elapsed_model_training,
            elapsed_application_to_encoded_data=elapsed_application_to_encoded_data,
            elapsed_refinement_and_reconstruction=elapsed_refinement_and_reconstruction,
            elapsed_total=elapsed_total,
            output_dir=current_experiment_directory
        )

    return 0

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dea_config.json")
    if not os.path.isfile(config_path):
        print(f"[ERROR] Could not find configuration file: {config_path}")
        sys.exit(1)
    with open(config_path, 'r') as f:
        configs = json.load(f)
    run_dea(
        configs.get('GLOBAL_CONFIG', {}),
        configs.get('ENC_CONFIG', {}),
        configs.get('EMB_CONFIG', {}),
        configs.get('ALIGN_CONFIG', {}),
        configs.get('DEA_CONFIG', {})
    )