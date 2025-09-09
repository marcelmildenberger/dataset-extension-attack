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
    estimate_pos_weight
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
        # Save termination log as CSV for better analysis
        termination_data = {
            "metric": ["Status", "Length of data_train", "Length of data_val", "Length of data_test"],
            "value": ["Training process canceled due to empty dataset", len(data_train), len(data_val), len(data_test)]
        }
        termination_df = pd.DataFrame(termination_data)
        termination_csv_path = os.path.join(current_experiment_directory, "termination_log.csv")
        termination_df.to_csv(termination_csv_path, index=False)
        raise Exception("Training process canceled due to empty dataset")

    # Start timing the hyperparameter optimization run.
    if GLOBAL_CONFIG["BenchMode"]:
        start_hyperparameter_optimization = time.time()

    # Define a function to train a model with a given configuration.
    # This function is used by Ray Tune to train models with different hyperparameters.
    def hyperparameter_training(config, data_dir, output_dim, alice_enc_hash, identifier, patience, min_delta, workers):
        # Resolve config
        batch_size       = int(config["batch_size"])
        num_layers       = config["num_layers"]
        hidden_layer_sz  = config["hidden_layer_size"]
        dropout_rate     = config["dropout_rate"]
        activation_fn    = config["activation_fn"]
        loss_cfg         = config["loss_fn"]
        opt_cfg          = config["optimizer"]
        sched_cfg        = config["lr_scheduler"]
        clip_grad_norm   = float(config.get("clip_grad_norm", 0.0))
        threshold        = config["threshold"]

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
            hidden_layer_size=hidden_layer_sz,
            dropout_rate=dropout_rate,
            activation_fn=activation_fn
        )
        model.to(device)

        # ----- Loss ----------------------------------------------------------------------------------
        if isinstance(loss_cfg, dict) and loss_cfg.get("name") == "BCEWithLogitsLoss":
            use_pw = bool(loss_cfg.get("use_pos_weight", False))
            if use_pw:
                # small single-pass estimate on the fly
                pos_weight = estimate_pos_weight(dataloader_train, output_dim).to(device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                criterion = nn.BCEWithLogitsLoss()
        elif isinstance(loss_cfg, dict) and loss_cfg.get("name") == "MultiLabelSoftMarginLoss":
            criterion = nn.MultiLabelSoftMarginLoss()
        elif isinstance(loss_cfg, dict) and loss_cfg.get("name") == "SoftMarginLoss":
            criterion = nn.SoftMarginLoss()

        # ----- Optimizer -----------------------------------------------------------------------------
        opt_name = opt_cfg["name"]
        lr = opt_cfg["lr"].sample()
        wd = opt_cfg["weight_decay"].sample()

        if opt_name == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=lr,
                                    alpha=opt_cfg["alpha"].sample(),
                                    weight_decay=wd)
        elif opt_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr,
                                momentum=opt_cfg["momentum"].sample(),
                                nesterov=bool(opt_cfg.get("nesterov", False)),
                                weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        # ----- Scheduler -----------------------------------------------------------------------------
        scheduler, scheduler_step = None, None  # scheduler_step in {"batch","epoch",None}

        name = sched_cfg["name"]
        if name == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=sched_cfg["step_size"].sample(),
                gamma=sched_cfg["gamma"].sample()
            )
            scheduler_step = "epoch"
        elif name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=sched_cfg["T_max"].sample(),
                eta_min=sched_cfg["eta_min"].sample()
            )
            scheduler_step = "epoch"
        elif name == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=sched_cfg.get("mode", "min"),
                factor=sched_cfg["factor"].sample(),
                patience=sched_cfg["patience"].sample(),
                min_lr=sched_cfg.get("min_lr", 0.0).sample()
            )
            scheduler_step = "plateau"  # special-case: step(metric) after val
        elif name == "CyclicLR":
            base_lr = sched_cfg["base_lr"].sample()
            ratio = sched_cfg["ratio"].sample()
            max_lr = base_lr * ratio
            # Keep optimizer LR aligned with base_lr for clarity:
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr
            cycle_momentum = isinstance(optimizer, optim.SGD) and optimizer.defaults.get("momentum", 0.0) > 0
            mode = sched_cfg.get("mode", "triangular")
            # If mode is a tune object, sample it
            if hasattr(mode, 'sample'):
                mode = mode.sample()
            
            # Handle exp_range mode which requires scale_fn
            if mode == "exp_range":
                scheduler = torch.optim.lr_scheduler.CyclicLR(
                    optimizer, base_lr=base_lr, max_lr=max_lr,
                    step_size_up=sched_cfg["step_size_up"].sample(),
                    mode=mode,
                    cycle_momentum=cycle_momentum,
                    scale_fn=lambda x: 0.8**(x)  # Exponential decay function
                )
            else:
                scheduler = torch.optim.lr_scheduler.CyclicLR(
                    optimizer, base_lr=base_lr, max_lr=max_lr,
                    step_size_up=sched_cfg["step_size_up"].sample(),
                    mode=mode,
                    cycle_momentum=cycle_momentum
                )
            scheduler_step = "batch"
        elif name == "None":
            scheduler = None
            scheduler_step = None
        else:
            raise ValueError(f"Unknown scheduler: {name}")

        # Training Loop

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
                is_training=True, verbose=GLOBAL_CONFIG["Verbose"], scheduler=scheduler, scheduler_step=scheduler_step, clip_grad_norm=clip_grad_norm
            )
            val_loss = run_epoch(
                model, dataloader_val, criterion, optimizer, device,
                is_training=False, verbose=GLOBAL_CONFIG["Verbose"], scheduler=None, scheduler_step=None  # never step during eval
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif scheduler is not None and scheduler_step == "epoch":
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
        "threshold": tune.uniform(0.3, 0.8),
        # MLP
        "num_layers": tune.choice([1, 2, 3]),
        "hidden_layer_size": tune.choice([256, 512, 1024, 2048, 4096]),
        "dropout_rate": tune.uniform(0.1, 0.5),
        "activation_fn": tune.choice(["relu", "leaky_relu", "gelu", "elu", "tanh"]),

        # Optimizer
        "optimizer": tune.choice([
            {"name": "AdamW", "lr": tune.loguniform(1e-5, 3e-3),
            "weight_decay": tune.loguniform(1e-6, 1e-2)},
            {"name": "Adam",  "lr": tune.loguniform(1e-5, 3e-3),
            "weight_decay": tune.loguniform(1e-7, 1e-3)},
            {"name": "RMSprop","lr": tune.loguniform(1e-5, 1e-3),
            "alpha": tune.uniform(0.8, 0.99), "weight_decay": tune.loguniform(1e-7, 1e-3)},
            {"name": "SGD",   "lr": tune.loguniform(1e-4, 1e-1),
            "momentum": tune.uniform(0.0, 0.95),
            "nesterov": tune.choice([False, True]),
            "weight_decay": tune.loguniform(1e-7, 1e-3)},
        ]),
        # Loss
        "loss_fn": tune.choice([
            {"name": "BCEWithLogitsLoss", "use_pos_weight": tune.choice([False, True])},
            {"name": "MultiLabelSoftMarginLoss"},
            {"name": "SoftMarginLoss"}
        ]),
        
        # LR scheduler
        "lr_scheduler": tune.choice([
            {"name": "None"},
            {"name": "StepLR", "step_size": tune.choice([5, 10, 20]), "gamma": tune.uniform(0.3, 0.8)},
            {"name": "CosineAnnealingLR", "T_max": tune.randint(10, 51), "eta_min": tune.choice([0.0, 1e-6, 1e-5])},
            {"name": "ReduceLROnPlateau", "mode": "min", "factor": tune.uniform(0.1, 0.5),
            "patience": tune.choice([3, 5, 10]), "min_lr": tune.choice([0.0, 1e-6, 1e-5])},
            {"name": "CyclicLR", "base_lr": tune.loguniform(1e-5, 1e-3),
            "ratio": tune.uniform(2.0, 10.0),
            "step_size_up": tune.choice([500, 1000, 2000]),
            "mode": tune.choice(["triangular", "triangular2", "exp_range"])},
        ]),
        # Batch size & regularization
        "batch_size": tune.choice([8, 16, 32, 64, 128]),
        "clip_grad_norm": tune.choice([0.0, 1.0, 5.0]),
    }

    # Initialize Ray for hyperparameter optimization.
    ray.init(
        num_cpus=GLOBAL_CONFIG["Workers"],
        num_gpus=2 if GLOBAL_CONFIG["UseGPU"] else 0,
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
        workers=GLOBAL_CONFIG["Workers"] // 11 if GLOBAL_CONFIG["UseGPU"] else 0,
    )

    # Wrap the trainable function with resources for 10 trials
    trainable_with_resources = tune.with_resources(
        trainable,
        resources={"cpu": GLOBAL_CONFIG["Workers"] // 11, "gpu": 0.18} if GLOBAL_CONFIG["UseGPU"] else {"cpu": GLOBAL_CONFIG["Workers"] // 11, "gpu": 0}
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
        print_and_save_result("best_result", best_result, hyperparameter_optimization_directory)

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
        num_workers=GLOBAL_CONFIG["Workers"] // 11 if GLOBAL_CONFIG["UseGPU"] else 0,
    )
    dataloader_val = DataLoader(
        data_val,
        batch_size=int(best_config.get("batch_size", 32)),
        shuffle=False,
        pin_memory=True,
        num_workers=GLOBAL_CONFIG["Workers"] // 11 if GLOBAL_CONFIG["UseGPU"] else 0,
    )
    dataloader_test = DataLoader(
        data_test,
        batch_size=int(best_config.get("batch_size", 32)),
        shuffle=False,
        pin_memory=True,
        num_workers=GLOBAL_CONFIG["Workers"] // 11 if GLOBAL_CONFIG["UseGPU"] else 0,
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

    loss_cfg = best_config.get("loss_fn")
    # Initialize the TensorBoard writer.
    if GLOBAL_CONFIG["SaveResults"]:
        run_name = "".join([
            loss_cfg.get("name", "MultiLabelSoftMarginLoss"),
            best_config.get("optimizer").get("name", "Adam"),
            ENC_CONFIG["AliceAlgo"],
            best_config.get("activation_fn", "relu"),
        ])
        tb_writer = SummaryWriter(f"{current_experiment_directory}/{run_name}")

        # ------------------- LOSS -------------------
    
    if isinstance(loss_cfg, dict):
        loss_name = loss_cfg.get("name", "MultiLabelSoftMarginLoss")
    else:
        loss_name = loss_cfg

    if loss_name == "BCEWithLogitsLoss":
        use_pw = isinstance(loss_cfg, dict) and loss_cfg.get("use_pos_weight", False)
        if use_pw:
            pos_weight = estimate_pos_weight(dataloader_train, len(all_two_grams)).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()
    elif loss_name == "MultiLabelSoftMarginLoss":
        criterion = nn.MultiLabelSoftMarginLoss()
    elif loss_name == "SoftMarginLoss":
        criterion = nn.SoftMarginLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")

    # ------------------- OPTIMIZER -------------------
    opt_cfg   = best_config.get("optimizer", {})
    opt_name  = opt_cfg.get("name", "Adam")
    lr        = opt_cfg.get("lr", 1e-3)
    weightdec = opt_cfg.get("weight_decay", 0.0)
    momentum  = opt_cfg.get("momentum", 0.0)
    nesterov  = opt_cfg.get("nesterov", False)
    alpha     = opt_cfg.get("alpha", 0.99)  # RMSprop smoothing

    if opt_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weightdec)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weightdec)
    elif opt_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, weight_decay=weightdec)
    elif opt_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                            nesterov=nesterov, weight_decay=weightdec)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    # ------------------- SCHEDULER -------------------
    sched_cfg = best_config.get("lr_scheduler", {"name": "None"})
    name = sched_cfg.get("name", "None")
    scheduler, scheduler_step = None, None  # {"batch","epoch",None}

    if name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sched_cfg.get("step_size", 10)),
            gamma=float(sched_cfg.get("gamma", 0.5))
        )
        scheduler_step = "epoch"

    elif name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(sched_cfg.get("T_max", 50)),
            eta_min=float(sched_cfg.get("eta_min", 0.0))
        )
        scheduler_step = "epoch"

    elif name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_cfg.get("mode", "min"),
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 5)),
            min_lr=float(sched_cfg.get("min_lr", 0.0))
        )
        scheduler_step = "plateau"  # call scheduler.step(val_loss) after validation

    elif name == "CyclicLR":
        # Prefer base_lr + ratio (max_lr = base_lr * ratio); fall back to explicit max_lr if provided
        base_lr = float(sched_cfg.get("base_lr", lr))
        ratio   = sched_cfg.get("ratio", None)
        max_lr  = float(base_lr * ratio) if ratio is not None else float(sched_cfg.get("max_lr", base_lr * 10.0))
        step_size_up = int(sched_cfg.get("step_size_up", 1000))
        mode = sched_cfg.get("mode", sched_cfg.get("mode_cyclic", "triangular"))

        # Align optimizer LR to base_lr for clarity
        for pg in optimizer.param_groups:
            pg["lr"] = base_lr

        cycle_momentum = isinstance(optimizer, optim.SGD) and momentum > 0.0

        if mode == "exp_range":
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=base_lr, max_lr=max_lr,
                step_size_up=step_size_up, mode=mode,
                cycle_momentum=cycle_momentum,
                scale_fn=lambda x: 0.8 ** x,
            )
        else:
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=base_lr, max_lr=max_lr,
                step_size_up=step_size_up, mode=mode,
                cycle_momentum=cycle_momentum,
            )
        scheduler_step = "batch"

    elif name in (None, "None"):
        scheduler = None
        scheduler_step = None

    else:
        raise ValueError(f"Unsupported LR scheduler: {name}")



    clip_grad_norm = best_config.get("clip_grad_norm", 0.0)

    def train_final_model(
    model,
    dataloader_train,
    dataloader_val,
    criterion,
    optimizer,
    device,
    scheduler=None,
    clip_grad_norm=0.0,
    scheduler_step=None,
    ):
        # epochs from best_config if present, otherwise default
        num_epochs = best_config.get("epochs", DEA_CONFIG["Epochs"])
        verbose    = GLOBAL_CONFIG["Verbose"]
        patience   = DEA_CONFIG["Patience"]
        min_delta  = DEA_CONFIG["MinDelta"]


        best_val_loss   = float("inf")
        best_model_state = None
        early_stopper   = EarlyStopping(patience=patience, min_delta=min_delta, verbose=verbose)
        train_losses, val_losses = [], []

        for epoch in range(num_epochs):
            # Train: if scheduler is per-batch, pass it in; otherwise don't
            train_loss = run_epoch(
                model, dataloader_train, criterion, optimizer, device,
                is_training=True, verbose=verbose,
                scheduler=scheduler,
                scheduler_step=scheduler_step,
                clip_grad_norm=clip_grad_norm,
            )
            train_losses.append(train_loss)

            # Validate: never step the scheduler inside validation
            val_loss = run_epoch(
                model, dataloader_val, criterion, optimizer, device,
                is_training=False, verbose=verbose,
                scheduler=None, scheduler_step=None,
                clip_grad_norm=0.0,
            )
            val_losses.append(val_loss)

            # Track best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif scheduler is not None and scheduler_step == "epoch":
                scheduler.step()

            log_epoch_metrics(
                epoch, num_epochs, train_loss, val_loss,
                tb_writer=tb_writer if 'tb_writer' in locals() else None,
                save_results=GLOBAL_CONFIG["SaveResults"]
            )

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
        scheduler=scheduler,
        clip_grad_norm=clip_grad_norm,
        scheduler_step=scheduler_step
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
    if GLOBAL_CONFIG["SaveModel"]:
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