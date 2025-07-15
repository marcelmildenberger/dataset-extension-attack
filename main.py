import argparse
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
    clean_result_dict,
    create_identifier_column_dynamic,
    create_optimized_dataloader,
    decode_labels_to_two_grams,
    filter_high_scoring_two_grams,
    fuzzy_reconstruction_approach,
    get_hashes,
    greedy_reconstruction,
    load_dataframe,
    lowercase_df,
    map_probabilities_to_two_grams,
    metrics_per_entry,
    print_and_save_result,
    read_header,
    reconstruct_identities_with_llm,
    reidentification_analysis,
    resolve_config,
    run_epoch,
    save_dea_runtime_log,
)

def run_dea(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG, DEA_CONFIG):
    ALIGN_CONFIG["RegWS"] = max(0.1, GLOBAL_CONFIG["Overlap"] / 3)
    GLOBAL_CONFIG["Workers"] = os.cpu_count() - 1
    alphabet = string.ascii_lowercase
    digits = string.digits
    letter_letter_grams = [a + b for a in alphabet for b in alphabet]
    digit_digit_grams = [d1 + d2 for d1 in digits for d2 in digits]
    letter_digit_grams = [l + d for l in alphabet for d in digits]
    all_two_grams = letter_letter_grams + letter_digit_grams + digit_digit_grams
    two_gram_dict = {i: two_gram for i, two_gram in enumerate(all_two_grams)}
    if GLOBAL_CONFIG["BenchMode"]:
        start_total = time.time()
        start_gma = time.time()
    data_dir = os.path.abspath("./data")
    eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash = get_hashes(
        GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG
    )
    identifier = f"{eve_enc_hash}_{alice_enc_hash}_{eve_emb_hash}_{alice_emb_hash}"
    path_reidentified = f"{data_dir}/available_to_eve/reidentified_individuals_{identifier}.h5"
    path_not_reidentified = f"{data_dir}/available_to_eve/not_reidentified_individuals_{identifier}.h5"
    path_all = f"{data_dir}/dev/alice_data_complete_with_encoding_{alice_enc_hash}.h5"
    if not (
        os.path.isfile(path_reidentified)
        and os.path.isfile(path_not_reidentified)
        and os.path.isfile(path_all)
    ):
        run_gma(
            GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG, DEA_CONFIG,
            eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash
        )
    if GLOBAL_CONFIG["BenchMode"]:
        elapsed_gma = time.time() - start_gma
    def get_cache_path(data_directory, identifier, alice_enc_hash, name="dataset"):
        os.makedirs(f"{data_directory}/cache", exist_ok=True)
        return os.path.join(data_directory, "cache", f"{name}_{identifier}_{alice_enc_hash}.pkl")
    def load_data(data_directory, alice_enc_hash, identifier, load_test=False):
        cache_path = get_cache_path(data_directory, identifier, alice_enc_hash)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                data_train, data_val, data_test = pickle.load(f)
            return data_train, data_val, data_test
        df_reidentified = load_dataframe(f"{data_directory}/available_to_eve/reidentified_individuals_{identifier}.h5")
        df_test = None
        if load_test:
            df_not_reidentified = load_dataframe(f"{data_directory}/available_to_eve/not_reidentified_individuals_{identifier}.h5")
            df_all = load_dataframe(f"{data_directory}/dev/alice_data_complete_with_encoding_{alice_enc_hash}.h5")
            df_test = df_all[df_all["uid"].isin(df_not_reidentified["uid"])].reset_index(drop=True)
        def get_encoding_dataset_class():
            algo = ENC_CONFIG["AliceAlgo"]
            if algo == "BloomFilter":
                return BloomFilterDataset
            elif algo == "TabMinHash":
                return TabMinHashDataset
            elif algo == "TwoStepHash":
                return TwoStepHashDataset
            else:
                raise ValueError(f"Unknown encoding algorithm: {algo}")
        DatasetClass = get_encoding_dataset_class()
        if ENC_CONFIG["AliceAlgo"] == "TwoStepHash":
            unique_ints = sorted(set().union(*df_reidentified["twostephash"]))
            dataset_args = {"all_integers": unique_ints}
        else:
            dataset_args = {}
        common_args = {
            "is_labeled": True,
            "all_two_grams": all_two_grams,
            "dev_mode": GLOBAL_CONFIG["DevMode"]
        }
        data_labeled = DatasetClass(df_reidentified, **common_args, **dataset_args)
        data_test = DatasetClass(df_test, **common_args, **dataset_args) if load_test else None
        train_size = int(DEA_CONFIG["TrainSize"] * len(data_labeled))
        val_size = len(data_labeled) - train_size
        data_train, data_val = random_split(data_labeled, [train_size, val_size])
        with open(cache_path, 'wb') as f:
            pickle.dump((data_train, data_val, data_test), f)
        return data_train, data_val, data_test
    def load_not_reidentified_data(data_directory, alice_enc_hash, identifier):
            cache_path = get_cache_path(data_directory, identifier, alice_enc_hash, name="not_reidentified")
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    df_filtered = pickle.load(f)
                return df_filtered
            df_not_reidentified = load_dataframe(f"{data_directory}/available_to_eve/not_reidentified_individuals_{identifier}.h5")
            df_all = load_dataframe(f"{data_directory}/dev/alice_data_complete_with_encoding_{alice_enc_hash}.h5")
            df_filtered = df_all[df_all["uid"].isin(df_not_reidentified["uid"])].reset_index(drop=True)
            drop_col = df_filtered.columns[-2]
            df_filtered = df_filtered.drop(columns=[drop_col])
            with open(cache_path, 'wb') as f:
                pickle.dump(df_filtered, f)
            return df_filtered
    selected_dataset = GLOBAL_CONFIG["Data"].split("/")[-1].replace(".tsv", "")
    experiment_tag = "experiment_" + ENC_CONFIG["AliceAlgo"] + "_" + selected_dataset + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_to = f"experiment_results/{experiment_tag}"
    os.makedirs(save_to, exist_ok=True)
    all_configs = {
        "GLOBAL_CONFIG": GLOBAL_CONFIG,
        "DEA_CONFIG": DEA_CONFIG,
        "ENC_CONFIG": ENC_CONFIG,
        "EMB_CONFIG": EMB_CONFIG,
        "ALIGN_CONFIG": ALIGN_CONFIG
    }
    with open(os.path.join(save_to, "config.txt"), "w") as f:
        for config_name, config_dict in all_configs.items():
            f.write(f"# === {config_name} ===\n")
            f.write(json.dumps(config_dict, indent=4))
            f.write("\n\n")
    os.makedirs(f"{save_to}/hyperparameteroptimization", exist_ok=True)
    data_train, data_val, data_test = load_data(data_dir, alice_enc_hash, identifier, load_test=True)
    df_not_reidentified = load_not_reidentified_data(data_dir, alice_enc_hash, identifier)
    if len(data_train) == 0 or len(data_val) == 0 or len(data_test) == 0 or df_not_reidentified.empty:
        log_path = os.path.join(save_to, "termination_log.txt")
        with open(log_path, "w") as f:
            f.write("Training process canceled due to empty dataset.\n")
            f.write(f"Length of data_train: {len(data_train)}\n")
            f.write(f"Length of data_val: {len(data_val)}\n")
            f.write(f"Length of data_test: {len(data_test)}\n")
            f.write(f"Length of df_not_reidentified: {len(df_not_reidentified)}\n")
        print("One or more datasets are empty. Termination log written.")
        return 0
    if GLOBAL_CONFIG["BenchMode"]:
        start_hyperparameter_optimization = time.time()
    # Optimization Potential: Yes. The code can be made more efficient, modular, and readable.
    # - Avoid repeated sampling of hyperparameters (sample() calls) by sampling once and reusing.
    # - Move device selection and model/loss/optimizer/scheduler creation outside the epoch loop.
    # - Use local variables for config access to reduce dict lookups.
    # - Use tqdm for progress bar if verbose.
    # - Avoid unnecessary deep copies if not needed.
    # - Use built-in Ray Tune features for reporting and checkpointing.
    # - Remove unused lists if not needed (train_losses, val_losses).
    # - Use dataloader.dataset length only if dataset implements __len__.
    # - Consider using DataLoader pin_memory and num_workers for speedup.

    def train_model(config, data_dir, output_dim, alice_enc_hash, identifier, patience, min_delta):
        # Sample all hyperparameters up front to avoid repeated .sample() calls
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
        data_train, data_val, _ = load_data(data_dir, alice_enc_hash, identifier, load_test=False)
        input_dim = data_train[0][0].shape[0]
        # Use optimized DataLoader configuration
        dataloader_train = create_optimized_dataloader(
            data_train,
            batch_size=batch_size,
            is_training=True
        )
        dataloader_val = create_optimized_dataloader(
            data_val,
            batch_size=batch_size,
            is_training=False
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = BaseModelHyperparameterOptimization(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
            activation_fn=activation_fn
        ).to(device)

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

        best_val_loss = float('inf')
        best_model_state = None
        total_precision = total_recall = total_f1 = total_dice = total_val_loss = 0.0
        num_samples = 0
        epochs = 0
        early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)

        for _ in range(DEA_CONFIG["Epochs"]):
            epochs += 1
            model.train()
            train_loss = run_epoch(
                model, dataloader_train, criterion, optimizer, device,
                is_training=True, verbose=GLOBAL_CONFIG["Verbose"], scheduler=scheduler
            )
            model.eval()
            val_loss = run_epoch(
                model, dataloader_val, criterion, optimizer, device,
                is_training=False, verbose=GLOBAL_CONFIG["Verbose"], scheduler=scheduler
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
            total_val_loss += val_loss
            if early_stopper(val_loss):
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
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

    ray.init(
        num_cpus=GLOBAL_CONFIG["Workers"],
        num_gpus=1 if GLOBAL_CONFIG["UseGPU"] else 0,
        ignore_reinit_error=True,
        logging_level="ERROR"
    )

    optuna_search = OptunaSearch(metric=DEA_CONFIG["MetricToOptimize"], mode="max")
    scheduler = ASHAScheduler(metric="total_val_loss", mode="min")

    trainable = partial(
        train_model,
        data_dir=data_dir,
        output_dim=len(all_two_grams),
        alice_enc_hash=alice_enc_hash,
        identifier=identifier,
        patience=DEA_CONFIG["Patience"],
        min_delta=DEA_CONFIG["MinDelta"]
    )

    # Each trial gets 1 CPU and 1 GPU (if available)
    trainable_with_resources = tune.with_resources(
        trainable,
        resources={"cpu": 1, "gpu": 1} if GLOBAL_CONFIG["UseGPU"] else {"cpu": 1, "gpu": 0}
    )

    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            scheduler=scheduler,
            num_samples=DEA_CONFIG["NumSamples"],
            max_concurrent_trials=GLOBAL_CONFIG["Workers"],
        ),
        param_space=search_space,
        run_config=air.RunConfig(name="dea_hpo_run")
    )
    results = tuner.fit()
    ray.shutdown()

    result_grid = results
    best_result = result_grid.get_best_result(metric=DEA_CONFIG["MetricToOptimize"], mode="max")
    if GLOBAL_CONFIG["BenchMode"]:
        elapsed_hyperparameter_optimization = time.time() - start_hyperparameter_optimization
    if GLOBAL_CONFIG["SaveResults"]:
        worst_result = result_grid.get_best_result(metric=DEA_CONFIG["MetricToOptimize"], mode="min")
        df = pd.DataFrame([
            {
                **clean_result_dict(resolve_config(result.config)),
                **{k: result.metrics.get(k) for k in ["average_dice", "average_precision", "average_recall", "average_f1"]},
            }
            for result in result_grid
        ])
        df.to_csv(f"{save_to}/hyperparameteroptimization/all_trial_results.csv", index=False)
        print("✅ Results saved to all_trial_results.csv")
        print_and_save_result("Best_Result", best_result, f"{save_to}/hyperparameteroptimization")
        print_and_save_result("Worst_Result", worst_result, f"{save_to}/hyperparameteroptimization")
        print("\n📊 Average Metrics Across All Trials")
        avg_metrics = df[["average_dice", "average_precision", "average_recall", "average_f1"]].mean()
        print("-" * 40)
        for key, value in avg_metrics.items():
            print(f"{key.capitalize()}: {value:.4f}")
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[["average_dice", "average_recall", "average_f1", "average_precision"]])
        plt.title("Distribution of Performance Metrics Across Trials")
        plt.grid(True)
        plt.savefig(f"{save_to}/hyperparameteroptimization/metric_distributions.png")
        plt.close()
        print("📊 Saved plot: metric_distributions.png")
        exclude_cols = {"input_dim", "output_dim"}
        numeric_config_cols = [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col]) and col not in exclude_cols
        ]
        correlation_df = df[numeric_config_cols].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_df, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Between Parameters and Metrics")
        plt.tight_layout()
        plt.savefig(f"{save_to}/hyperparameteroptimization/correlation_heatmap.png")
        plt.close()
        print("📌 Saved heatmap: correlation_heatmap.png")
    if GLOBAL_CONFIG["BenchMode"]:
        start_model_training = time.time()
    best_config = resolve_config(best_result.config)
    data_train, data_val, data_test = load_data(data_dir, alice_enc_hash, identifier, load_test=True)
    input_dim=data_train[0][0].shape[0]
    # Use optimized DataLoader configuration
    batch_size = int(best_config.get("batch_size", 32))
    dataloader_train = create_optimized_dataloader(
        data_train,
        batch_size=batch_size,
        is_training=True
    )
    dataloader_val = create_optimized_dataloader(
        data_val,
        batch_size=batch_size,
        is_training=False
    )
    dataloader_test = create_optimized_dataloader(
        data_test,
        batch_size=batch_size,
        is_training=False
    )
    model = BaseModel(
                input_dim=input_dim,
                output_dim=len(all_two_grams),
                hidden_layer=best_config.get("hidden_layer_size", 128),
                num_layers=best_config.get("num_layers", 2),
                dropout_rate=best_config.get("dropout_rate", 0.2),
                activation_fn=best_config.get("activation_fn", "relu")
            )
    print(model)
    if GLOBAL_CONFIG["SaveResults"]:
        run_name = "".join([
            best_config.get("loss_fn", "MultiLabelSoftMarginLoss"),
            best_config.get("optimizer").get("name", "Adam"),
            ENC_CONFIG["AliceAlgo"],
            best_config.get("activation_fn", "relu"),
        ])
        tb_writer = SummaryWriter(f"{save_to}/{run_name}")
    compute_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(compute_device)
    match best_config.get("loss_fn", "MultiLabelSoftMarginLoss"):
        case "BCEWithLogitsLoss":
            criterion = nn.BCEWithLogitsLoss(reduction='mean')
        case "MultiLabelSoftMarginLoss":
            criterion = nn.MultiLabelSoftMarginLoss(reduction='mean')
        case "SoftMarginLoss":
            criterion = nn.SoftMarginLoss()
        case _:
            raise ValueError(f"Unsupported loss function: {best_config.get('loss_fn', 'MultiLabelSoftMarginLoss')}")
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
    def _log_epoch_metrics(epoch, total_epochs, train_loss, val_loss):
        epoch_str = f"[{epoch + 1}/{total_epochs}]"
        print(f"{epoch_str} 🔧 Train Loss: {train_loss:.4f} | 🔍 Val Loss: {val_loss:.4f}")
        if DEA_CONFIG.get("SaveResults", False) and 'tb_writer' in globals():
            tb_writer.add_scalar("Loss/train", train_loss, epoch + 1)
            tb_writer.add_scalar("Loss/validation", val_loss, epoch + 1)
    def train_model(model, dataloader_train, dataloader_val, criterion, optimizer, device, scheduler=None):
        num_epochs = best_config.get("epochs", DEA_CONFIG["Epochs"])
        verbose = GLOBAL_CONFIG["Verbose"]
        patience = DEA_CONFIG["Patience"]
        min_delta = DEA_CONFIG["MinDelta"]
        best_val_loss = float('inf')
        best_model_state = None
        early_stopper = EarlyStopping(patience=patience, min_delta=min_delta, verbose=verbose)
        train_losses, val_losses = [], []
        for epoch in range(num_epochs):
            model.train()
            train_loss = run_epoch(
                model, dataloader_train, criterion, optimizer,
                device, is_training=True, verbose=verbose, scheduler=scheduler
            )
            train_losses.append(train_loss)
            model.eval()
            with torch.no_grad():
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
            _log_epoch_metrics(epoch, num_epochs, train_loss, val_loss)
            if early_stopper(val_loss):
                if verbose:
                    print(f"⏹️ Early stopping triggered at epoch {epoch + 1}")
                break
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        return model, train_losses, val_losses
    model, train_losses, val_losses = train_model(
        model, dataloader_train, dataloader_val,
        criterion, optimizer, compute_device,
        scheduler=scheduler
    )
    if GLOBAL_CONFIG["BenchMode"]:
        elapsed_model_training = time.time() - start_model_training
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training loss', color='blue')
    plt.plot(val_losses, label='Validation loss', color='red')
    plt.legend()
    plt.title("Training and Validation Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    if DEA_CONFIG.get("SaveResults", False):
        plt.savefig(f"{save_to}/loss_curve.png")
        plt.close()
        print("📉 Saved loss curve to loss_curve.png")
    base_path = os.path.join(
        GLOBAL_CONFIG["LoadPath"] if GLOBAL_CONFIG["LoadResults"] else save_to,
        "trained_model"
    )
    model_file   = f"{base_path}/model.pt"
    config_file  = f"{base_path}/config.json"
    result_file  = f"{base_path}/result.json"
    metrics_file = f"{base_path}/metrics.txt"
    def save_model_and_config(model, config, path):
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, "model.pt"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        print(f"✅ Saved model and config to {path}")
    def load_model_and_config(model_cls, path, input_dim, output_dim):
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        model = model_cls(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layer=config.get("hidden_layer_size", 128),
            num_layers=config.get("num_layers", 2),
            dropout_rate=config.get("dropout_rate", 0.2),
            activation_fn=config.get("activation_fn", "relu")
        )
        model.load_state_dict(torch.load(os.path.join(path, "model.pt")))
        model.eval()
        return model, config
    if GLOBAL_CONFIG["SaveResults"]:
        save_model_and_config(model, best_config, base_path)
    if GLOBAL_CONFIG["LoadResults"]:
        model, best_config = load_model_and_config(BaseModel, base_path, input_dim=1024, output_dim=len(all_two_grams))
        model.to(compute_device)
    if GLOBAL_CONFIG["BenchMode"]:
        start_application_to_encoded_data = time.time()
    total_dice = total_precision = total_recall = total_f1 = 0.0
    num_samples = 0
    results = []
    threshold = best_config.get("threshold", 0.5)
    model.eval()
    dataloader_iter = tqdm(dataloader_test, desc="Test loop") if GLOBAL_CONFIG["Verbose"] else dataloader_test
    with torch.no_grad():
        for data, labels, uids in dataloader_iter:
            data, labels = data.to(compute_device), labels.to(compute_device)
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
    if num_samples > 0:
        avg_dice = total_dice / num_samples
        avg_precision = total_precision / num_samples
        avg_recall = total_recall / num_samples
        avg_f1 = total_f1 / num_samples
    else:
        avg_dice = avg_precision = avg_recall = avg_f1 = 0.0
    print(f"\n📊 Final Test Metrics:")
    print(f"  Dice:      {avg_dice:.4f}")
    print(f"  Precision: {avg_precision:.4f}")
    print(f"  Recall:    {avg_recall:.4f}")
    print(f"  F1 Score:  {avg_f1:.4f}")
    if GLOBAL_CONFIG["BenchMode"]:
        elapsed_application_to_encoded_data = time.time() - start_application_to_encoded_data
    if GLOBAL_CONFIG["SaveResults"]:
        with open(metrics_file, "w") as f:
            f.write(f"Average Precision: {avg_precision:.4f}\n")
            f.write(f"Average Recall: {avg_recall:.4f}\n")
            f.write(f"Average F1 Score: {avg_f1:.4f}\n")
            f.write(f"Average Dice Similarity: {avg_dice:.4f}\n")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
    if GLOBAL_CONFIG["LoadResults"]:
        with open(result_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
    results_df = pd.json_normalize(results)
    metric_cols = ["precision", "recall", "f1", "dice", "jaccard"]
    melted = results_df.melt(value_vars=metric_cols,
                            var_name="metric",
                            value_name="score")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=melted,
                x="score",
                hue="metric",
                bins=20,
                element="step",
                fill=False,
                kde=True,
                palette="Set2")
    plt.title("Distribution of Precision / Recall / F1 across Samples")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    if DEA_CONFIG.get("SaveResults", False):
        plt.savefig(f"{save_to}/metric_distributions.png")
        print("📊  Saved plot: metric_distributions.png")
    plt.close()
    print("\n🔍 Sample Reconstructions (first 5)")
    for _, row in results_df.iloc[:5].iterrows():
        print(f"UID: {row.uid}")
        print(f"  Actual 2-grams:    {row.actual_two_grams}")
        print(f"  Predicted 2-grams: {row.predicted_two_grams}")
        print("-" * 60)
    if GLOBAL_CONFIG["BenchMode"]:
        start_refinement_and_reconstruction = time.time()
    @lru_cache(maxsize=None)
    def get_not_reidentified_df(data_dir: str, identifier: str) -> pd.DataFrame:
        df = load_not_reidentified_data(data_dir, alice_enc_hash, identifier)
        return lowercase_df(df)
    def create_identifier(df: pd.DataFrame, comps):
        df = df.copy()
        df["identifier"] = create_identifier_column_dynamic(df, comps)
        return df[["uid", "identifier"]]
    def run_reidentification_once(reconstructed, df_not_reidentified, merge_cols, technique, identifier_components=None):
        df_reconstructed = lowercase_df(pd.DataFrame(reconstructed, columns=merge_cols))
        if(identifier_components):
            df_not_reidentified = create_identifier(df_not_reidentified, identifier_components)
        return reidentification_analysis(
            df_reconstructed,
            df_not_reidentified,
            merge_cols,
            len(df_not_reidentified),
            technique,
            save_path=f"{save_to}/re_identification_results"
        )
    header = read_header(GLOBAL_CONFIG["Data"])
    include_birthday = not (GLOBAL_CONFIG["Data"] == "./data/datasets/titanic_full.tsv")
    TECHNIQUES = {
        "ai": {
            "fn": reconstruct_identities_with_llm,
            "merge_cols": header[:3] + [header[-1]],
            "identifier_comps": None,
        },
        "greedy": {
            "fn": greedy_reconstruction,
            "merge_cols": ["uid", "identifier"],
            "identifier_comps": header[:-1],
        },
        "fuzzy": {
            "fn": fuzzy_reconstruction_approach,
            "merge_cols": (header[:3] if include_birthday else header[:2]) + [header[-1]],
            "identifier_comps": None,
        },
    }
    selected = DEA_CONFIG["MatchingTechnique"]
    df_not_reid_cached = get_not_reidentified_df(data_dir, identifier)
    save_dir = f"{save_to}/re_identification_results"
    if selected == "fuzzy_and_greedy":
        reidentified = {}
        for name in ("greedy", "fuzzy"):
            info = TECHNIQUES[name]
            if name == "fuzzy":
                recon = info["fn"](results, GLOBAL_CONFIG["Workers"], include_birthday )
            else:
                recon = info["fn"](results)
            reidentified[name] = run_reidentification_once(
                recon,
                df_not_reid_cached,
                info["merge_cols"],
                name,
                info["identifier_comps"],
            )
    else:
        if selected not in TECHNIQUES:
            raise ValueError(f"Unsupported matching technique: {selected}")
        info = TECHNIQUES[selected]
        if selected == "fuzzy":
            recon = info["fn"](results, GLOBAL_CONFIG["Workers"], include_birthday)
        if selected == "ai":
            recon = info["fn"](results, info["merge_cols"][:-1])
        else:
            recon = info["fn"](results)
        reidentified = run_reidentification_once(
            recon,
            df_not_reid_cached,
            info["merge_cols"],
            selected,
            info["identifier_comps"],
        )
    if selected == "fuzzy_and_greedy":
        uids_greedy = set(reidentified["greedy"]["uid"])
        uids_fuzzy = set(reidentified["fuzzy"]["uid"])
        combined_uids = uids_greedy.union(uids_fuzzy)
        total_reidentified_combined = len(combined_uids)
        df_not_reid_cached = get_not_reidentified_df(data_dir, identifier)
        len_not_reidentified = len(df_not_reid_cached)
        reidentification_rate_combined = (total_reidentified_combined / len_not_reidentified) * 100
        print("\n🔁 Combined Reidentification (greedy ∪ fuzzy):")
        print(f"Total not re-identified individuals: {len_not_reidentified}")
        print(f"Total Unique Reidentified Individuals: {total_reidentified_combined}")
        print(f"Combined Reidentification Rate: {reidentification_rate_combined:.2f}%")
        save_dir = os.path.join(save_to, "re_identification_results")
        os.makedirs(save_dir, exist_ok=True)
        pd.DataFrame({"uid": list(combined_uids)}).to_csv(
            os.path.join(save_dir, "result_fuzzy_and_greedy.csv"),
            index=False
        )
        summary_path = os.path.join(save_dir, "summary_fuzzy_and_greedy.txt")
        with open(summary_path, "w") as f:
            f.write("Reidentification Method: fuzzy_and_greedy\n")
            f.write(f"Total not re-identified individuals: {len_not_reidentified}\n")
            f.write(f"Total Unique Reidentified Individuals: {total_reidentified_combined}\n")
            f.write(f"Combined Reidentification Rate: {reidentification_rate_combined:.2f}%\n")
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
            output_dir=save_to
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