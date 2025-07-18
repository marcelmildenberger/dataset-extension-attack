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
    # TODO: Change saving config to json instead of txt
    selected_dataset = GLOBAL_CONFIG["Data"].split("/")[-1].replace(".tsv", "")
    experiment_tag = "experiment_" + ENC_CONFIG["AliceAlgo"] + "_" + selected_dataset + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_experiment_directory = f"/mnt/nvme/experiment_results/{experiment_tag}"
    os.makedirs(current_experiment_directory, exist_ok=True)
    all_configs = {
        "GLOBAL_CONFIG": GLOBAL_CONFIG,
        "DEA_CONFIG": DEA_CONFIG,
        "ENC_CONFIG": ENC_CONFIG,
        "EMB_CONFIG": EMB_CONFIG,
        "ALIGN_CONFIG": ALIGN_CONFIG
    }
    with open(os.path.join(current_experiment_directory, "config.txt"), "w") as f:
        for config_name, config_dict in all_configs.items():
            f.write(f"# === {config_name} ===\n")
            f.write(json.dumps(config_dict, indent=4))
            f.write("\n\n")

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
        start_gma = time.time()

    # Get the hashes for the encoding and embedding.
    eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash = get_hashes(
        GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG
    )

    # Get the data directory and identifier for the current run to check if the data is already available.
    data_dir = os.path.abspath("./data")
    identifier = f"{eve_enc_hash}_{alice_enc_hash}_{eve_emb_hash}_{alice_emb_hash}"
    path_reidentified = f"{data_dir}/available_to_eve/reidentified_individuals_{identifier}.h5"
    path_not_reidentified = f"{data_dir}/available_to_eve/not_reidentified_individuals_{identifier}.h5"
    path_all = f"{data_dir}/dev/alice_data_complete_with_encoding_{alice_enc_hash}.h5"

    # If the data is not available, run the GMA to generate it.
    if not (
        os.path.isfile(path_reidentified)
        and os.path.isfile(path_not_reidentified)
        and os.path.isfile(path_all)
    ):
        run_gma(
            GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG, DEA_CONFIG,
            eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash
        )

    # If the data is available, log the time taken for the GMA run.
    if GLOBAL_CONFIG["BenchMode"]:
        elapsed_gma = time.time() - start_gma

    # Load the experiment datasets (train, val, test) and check for empty splits.
    # If any split is empty, write a termination log and exit early to prevent downstream errors.
    datasets = load_experiment_datasets(data_dir, alice_enc_hash, identifier, ENC_CONFIG, DEA_CONFIG, GLOBAL_CONFIG, all_two_grams, splits=("train", "val", "test"))
    data_train, data_val, data_test = datasets["train"], datasets["val"], datasets["test"]
    if len(data_train) == 0 or len(data_val) == 0 or len(data_test) == 0:
        log_path = os.path.join(current_experiment_directory, "termination_log.txt")
        with open(log_path, "w") as f:
            f.write("Training process canceled due to empty dataset.\n")
            f.write(f"Length of data_train: {len(data_train)}\n")
            f.write(f"Length of data_val: {len(data_val)}\n")
            f.write(f"Length of data_test: {len(data_test)}\n")
        raise ValueError(f"Empty dataset: train={len(data_train)}, val={len(data_val)}, test={len(data_test)}")

    # Start timing the hyperparameter optimization run.
    if GLOBAL_CONFIG["BenchMode"]:
        start_hyperparameter_optimization = time.time()

    # Define a function to train a model with a given configuration.
    # This function is used by Ray Tune to train models with different hyperparameters.
    def hyperparameter_training(config, data_dir, output_dim, alice_enc_hash, identifier, patience, min_delta, workers):
        # Sample all hyperparameters up front
        batch_size = int(config["batch_size"])
        num_layers = config["num_layers"]
        hidden_layer_size = config["hidden_layer_size"]
        dropout_rate = config["dropout_rate"]
        activation_fn = config["activation_fn"]
        loss_fn_name = config["loss_fn"]
        threshold = config["threshold"]
        optimizer_cfg = config["optimizer"]
        lr_scheduler_cfg = config["lr_scheduler"]

        # Load data
        datasets = load_experiment_datasets(data_dir, alice_enc_hash, identifier, ENC_CONFIG, DEA_CONFIG, GLOBAL_CONFIG, all_two_grams, splits=("train", "val"))
        data_train, data_val = datasets["train"], datasets["val"]
        input_dim = data_train[0][0].shape[0]

        dataloader_train = DataLoader(
            data_train,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=workers,
        )
        dataloader_val = DataLoader(
            data_val,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=workers,
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = BaseModelHyperparameterOptimization(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
            activation_fn=activation_fn
        )
        model.to(device)

        # Loss function
        loss_functions = {
            "BCEWithLogitsLoss": nn.BCEWithLogitsLoss(),
            "MultiLabelSoftMarginLoss": nn.MultiLabelSoftMarginLoss(),
            "SoftMarginLoss": nn.SoftMarginLoss(),
        }
        criterion = loss_functions[loss_fn_name]

        # Optimizer
        lr = optimizer_cfg["lr"].sample() if hasattr(optimizer_cfg["lr"], "sample") else optimizer_cfg["lr"]
        optimizer_name = optimizer_cfg["name"]
        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=lr)
        elif optimizer_name == "SGD":
            momentum = optimizer_cfg.get("momentum", 0.0)
            if hasattr(momentum, "sample"):
                momentum = momentum.sample()
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        elif optimizer_name == "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Scheduler
        scheduler = None
        scheduler_name = lr_scheduler_cfg["name"]
        if scheduler_name == "StepLR":
            step_size = lr_scheduler_cfg["step_size"].sample() if hasattr(lr_scheduler_cfg["step_size"], "sample") else lr_scheduler_cfg["step_size"]
            gamma = lr_scheduler_cfg["gamma"].sample() if hasattr(lr_scheduler_cfg["gamma"], "sample") else lr_scheduler_cfg["gamma"]
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == "ExponentialLR":
            gamma = lr_scheduler_cfg["gamma"].sample() if hasattr(lr_scheduler_cfg["gamma"], "sample") else lr_scheduler_cfg["gamma"]
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif scheduler_name == "ReduceLROnPlateau":
            factor = lr_scheduler_cfg["factor"].sample() if hasattr(lr_scheduler_cfg["factor"], "sample") else lr_scheduler_cfg["factor"]
            patience_sched = lr_scheduler_cfg["patience"].sample() if hasattr(lr_scheduler_cfg["patience"], "sample") else lr_scheduler_cfg["patience"]
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=lr_scheduler_cfg["mode"],
                factor=factor,
                patience=patience_sched
            )
        elif scheduler_name == "CosineAnnealingLR":
            T_max = lr_scheduler_cfg["T_max"].sample() if hasattr(lr_scheduler_cfg["T_max"], "sample") else lr_scheduler_cfg["T_max"]
            eta_min = lr_scheduler_cfg["eta_min"].sample() if hasattr(lr_scheduler_cfg["eta_min"], "sample") else lr_scheduler_cfg["eta_min"]
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        elif scheduler_name == "CyclicLR":
            base_lr = lr_scheduler_cfg["base_lr"].sample() if hasattr(lr_scheduler_cfg["base_lr"], "sample") else lr_scheduler_cfg["base_lr"]
            max_lr = lr_scheduler_cfg["max_lr"].sample() if hasattr(lr_scheduler_cfg["max_lr"], "sample") else lr_scheduler_cfg["max_lr"]
            step_size_up = lr_scheduler_cfg["step_size_up"].sample() if hasattr(lr_scheduler_cfg["step_size_up"], "sample") else lr_scheduler_cfg["step_size_up"]
            mode_cyclic = lr_scheduler_cfg["mode_cyclic"].sample() if hasattr(lr_scheduler_cfg["mode_cyclic"], "sample") else lr_scheduler_cfg["mode_cyclic"]
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=base_lr,
                max_lr=max_lr,
                step_size_up=step_size_up,
                mode=mode_cyclic,
                cycle_momentum=False
            )
        elif scheduler_name == "None":
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

        # Initialize variables for tracking best validation loss, model state, performance metrics, and early stopping
        best_val_loss = float('inf')
        best_model_state = None
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_dice = 0.0
        total_val_loss = 0.0
        num_samples = 0
        epochs = 0
        early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)

        # Train the model for a specified number of epochs.
        for _ in range(DEA_CONFIG["Epochs"]):
            epochs += 1
            train_loss = run_epoch(
                model, dataloader_train, criterion, optimizer, device,
                is_training=True, verbose=GLOBAL_CONFIG["Verbose"], scheduler=scheduler
            )
            val_loss = run_epoch(
                model, dataloader_val, criterion, optimizer, device,
                is_training=False, verbose=GLOBAL_CONFIG["Verbose"], scheduler=scheduler
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif scheduler is not None:
                scheduler.step()
            total_val_loss += val_loss
            if early_stopper(val_loss):
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        for data, labels, _ in dataloader_val:
            actual_two_grams = decode_labels_to_two_grams(two_gram_dict, labels)
            data = data.to(device)
            logits = model(data)
            probabilities = torch.sigmoid(logits)
            batch_two_gram_scores = map_probabilities_to_two_grams(two_gram_dict, probabilities)
            batch_filtered_two_gram_scores = filter_high_scoring_two_grams(batch_two_gram_scores, threshold)
            dice, precision, recall, f1 = calculate_performance_metrics(
                actual_two_grams, batch_filtered_two_gram_scores)
            total_dice += dice
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            num_samples += data.size(0)
        train.report({
                "average_dice": total_dice / num_samples,
                "average_precision": total_precision / num_samples,
                "average_recall": total_recall / num_samples,
                "average_f1": total_f1 / num_samples,
                "total_val_loss": total_val_loss,
                "len_train": len(dataloader_train.dataset),
                "len_val": len(dataloader_val.dataset),
                "epochs": epochs
            })

    # Define the search space for the hyperparameters.
    search_space = {
        "output_dim": len(all_two_grams),
        "num_layers": tune.randint(1, 4),
        "hidden_layer_size": tune.choice([128, 256, 512, 1024, 2048]),
        "dropout_rate": tune.uniform(0.1, 0.4),
        "activation_fn": tune.choice(["relu", "leaky_relu", "gelu", "elu", "selu", "tanh"]),
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
            {"name": "CosineAnnealingLR", "T_max": tune.loguniform(10, 50), "eta_min": tune.choice([1e-5, 1e-6, 0])},
            {"name": "CyclicLR", "base_lr": tune.loguniform(1e-5, 1e-3), "max_lr": tune.loguniform(1e-3, 1e-1), "step_size_up": tune.choice([2000, 4000]), "mode_cyclic": tune.choice(["triangular", "triangular2", "exp_range"]) },
            {"name": "None"}
        ]),
        "batch_size": tune.choice([8, 16, 32, 64]),
    }

    # Initialize Ray for hyperparameter optimization.
    ray.init(
        num_cpus=GLOBAL_CONFIG["Workers"],
        num_gpus=1 if GLOBAL_CONFIG["UseGPU"] else 0,
        ignore_reinit_error=True,
        logging_level="ERROR"
    )

    # Initialize the Optuna search and the ASHAScheduler.
    optuna_search = OptunaSearch(metric=DEA_CONFIG["MetricToOptimize"], mode="max")
    scheduler = ASHAScheduler(metric="total_val_loss", mode="min")

    # Define the trainable function.
    trainable = partial(
        hyperparameter_training,
        data_dir=data_dir,
        output_dim=len(all_two_grams),
        alice_enc_hash=alice_enc_hash,
        identifier=identifier,
        patience=DEA_CONFIG["Patience"],
        min_delta=DEA_CONFIG["MinDelta"],
        workers=GLOBAL_CONFIG["Workers"] // 10 if GLOBAL_CONFIG["UseGPU"] else 0,
    )

    # Wrap the trainable function with resources for 10 trials
    trainable_with_resources = tune.with_resources(
        trainable,
        resources={"cpu": GLOBAL_CONFIG["Workers"] // 10, "gpu": 0.1} if GLOBAL_CONFIG["UseGPU"] else {"cpu": GLOBAL_CONFIG["Workers"]//6, "gpu": 0}
    )

    # Initialize the tuner.
    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            scheduler=scheduler,
            num_samples=DEA_CONFIG["NumSamples"],
        ),
        param_space=search_space,
        run_config=air.RunConfig(name="dea_hpo_run")
    )

    # Run the hyperparameter optimization.
    results = tuner.fit()

    # Shut down Ray.
    ray.shutdown()

    # Stop timing the hyperparameter optimization.
    if GLOBAL_CONFIG["BenchMode"]:
        elapsed_hyperparameter_optimization = time.time() - start_hyperparameter_optimization

    # Save the results of hyperparameter optimization to a CSV file and a plot.
    hyperparameter_optimization_directory = f"{current_experiment_directory}/hyperparameteroptimization"
    os.makedirs(hyperparameter_optimization_directory, exist_ok=True)

    # Get the best result.
    best_result = results.get_best_result(metric=DEA_CONFIG["MetricToOptimize"], mode="max")
    if GLOBAL_CONFIG["SaveResults"]:
        print_and_save_result("Best Result", best_result, hyperparameter_optimization_directory)

    # Start timing the model training.
    if GLOBAL_CONFIG["BenchMode"]:
        start_model_training = time.time()

    best_config = resolve_config(best_result.config)
    datasets = load_experiment_datasets(data_dir, alice_enc_hash, identifier, ENC_CONFIG, DEA_CONFIG, GLOBAL_CONFIG, all_two_grams, splits=("train", "val", "test"))
    data_train, data_val, data_test = datasets["train"], datasets["val"], datasets["test"]

    input_dim=data_train[0][0].shape[0]
    dataloader_train = DataLoader(
        data_train,
        batch_size=int(best_config.get("batch_size", 32)),
        shuffle=True,
        pin_memory=True,
        num_workers=GLOBAL_CONFIG["Workers"] // 10 if GLOBAL_CONFIG["UseGPU"] else 0,
    )
    dataloader_val = DataLoader(
        data_val,
        batch_size=int(best_config.get("batch_size", 32)),
        shuffle=False,
        pin_memory=True,
        num_workers=GLOBAL_CONFIG["Workers"] // 10 if GLOBAL_CONFIG["UseGPU"] else 0,
    )
    dataloader_test = DataLoader(
        data_test,
        batch_size=int(best_config.get("batch_size", 32)),
        shuffle=False,
        pin_memory=True,
        num_workers=GLOBAL_CONFIG["Workers"] // 10 if GLOBAL_CONFIG["UseGPU"] else 0,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BaseModel(
                input_dim=input_dim,
                output_dim=len(all_two_grams),
                hidden_layer=best_config.get("hidden_layer_size", 128),
                num_layers=best_config.get("num_layers", 2),
                dropout_rate=best_config.get("dropout_rate", 0.2),
                activation_fn=best_config.get("activation_fn", "relu")
            )
    model.to(device)

    # Initialize the TensorBoard writer.
    if GLOBAL_CONFIG["SaveResults"]:
        run_name = "".join([
            best_config.get("loss_fn", "MultiLabelSoftMarginLoss"),
            best_config.get("optimizer").get("name", "Adam"),
            ENC_CONFIG["AliceAlgo"],
            best_config.get("activation_fn", "relu"),
        ])
        tb_writer = SummaryWriter(f"{current_experiment_directory}/{run_name}")

    # Initialize the loss function.
    match best_config.get("loss_fn", "MultiLabelSoftMarginLoss"):
        case "BCEWithLogitsLoss":
            criterion = nn.BCEWithLogitsLoss(reduction='mean')
        case "MultiLabelSoftMarginLoss":
            criterion = nn.MultiLabelSoftMarginLoss(reduction='mean')
        case "SoftMarginLoss":
            criterion = nn.SoftMarginLoss()
        case _:
            raise ValueError(f"Unsupported loss function: {best_config.get('loss_fn', 'MultiLabelSoftMarginLoss')}")

    # Initialize the optimizer.
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

    # Initialize the learning rate scheduler.
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
                cycle_momentum=False
            )
        case None | "None":
            scheduler = None
        case _:
            raise ValueError(f"Unsupported LR scheduler: {best_config.get('lr_scheduler').get('name', 'None')}")

    def train_final_model(model, dataloader_train, dataloader_val, criterion, optimizer, device, scheduler=None):
        num_epochs = best_config.get("epochs", DEA_CONFIG["Epochs"])
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

            log_epoch_metrics(epoch, num_epochs, train_loss, val_loss, tb_writer=tb_writer if 'tb_writer' in locals() else None, save_results=GLOBAL_CONFIG["SaveResults"])
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



    # Save the trained model and config if requested.
    if GLOBAL_CONFIG["SaveResults"]:
        torch.save(model.state_dict(), os.path.join(trained_model_directory, "model.pt"))
        with open(os.path.join(trained_model_directory, "config.json"), "w") as f:
            json.dump(best_config, f, indent=4)

    # Start timing the application to encoded data.
    if GLOBAL_CONFIG["BenchMode"]:
        start_application_to_encoded_data = time.time()

    # Initialize the metrics.
    total_dice = total_precision = total_recall = total_f1 = 0.0
    num_samples = 0

    # Initialize the results.
    results = []

    # Initialize the threshold.
    threshold = best_config.get("threshold", 0.5)

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
        with open(f"{trained_model_directory}/metrics.txt", "w") as f:
            f.write(f"Average Precision: {avg_precision:.4f}\n")
            f.write(f"Average Recall: {avg_recall:.4f}\n")
            f.write(f"Average F1 Score: {avg_f1:.4f}\n")
            f.write(f"Average Dice Similarity: {avg_dice:.4f}\n")
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
    df_not_reid_cached = get_not_reidentified_df(data_dir, identifier, alice_enc_hash=alice_enc_hash)
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
            elapsed_gma=elapsed_gma,
            elapsed_hyperparameter_optimization=elapsed_hyperparameter_optimization,
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