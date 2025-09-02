import os
import json
import re
import pandas as pd
from pathlib import Path
import ast

def parse_best_result_json(path):
    """Parse the best result from hyperparameter optimization JSON file"""
    try:
        with open(path, 'r') as f:
            result = json.load(f)
        return result
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def parse_config_json(path):
    """Parse the experiment configuration from JSON file"""
    try:
        with open(path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def parse_metrics_csv(path):
    """Parse the trained model metrics from CSV file"""
    try:
        df = pd.read_csv(path)
        metrics = {}
        for _, row in df.iterrows():
            metric_name = row['metric']
            value = row['value']
            # Convert numeric values
            try:
                if isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                    value = float(value)
            except:
                pass
            metrics[metric_name] = value
        return metrics
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def parse_summary_csv(path):
    """Parse the reidentification summary from CSV file"""
    try:
        df = pd.read_csv(path)
        # Look for reidentification rate
        for _, row in df.iterrows():
            if 'reidentification_rate' in row['metric'].lower():
                value = row['value']
                if isinstance(value, str) and '%' in value:
                    # Extract numeric value from percentage string
                    rate_str = value.replace('%', '').strip()
                    try:
                        return float(rate_str) / 100
                    except:
                        pass
        return None
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def parse_runtime_csv(path):
    """Parse the DEA runtime log from CSV file"""
    try:
        df = pd.read_csv(path)
        runtimes = {}
        for _, row in df.iterrows():
            phase = row['phase']
            # Convert phase names to match expected format
            if phase == "gma":
                key = "GraphMatchingAttack"
            elif phase == "hyperparameter_optimization":
                key = "HyperparameterOptimization"
            elif phase == "model_training":
                key = "ModelTraining"
            elif phase == "application_to_encoded_data":
                key = "ApplicationtoEncodedData"
            elif phase == "refinement_and_reconstruction":
                key = "RefinementandReconstruction"
            elif phase == "total_runtime":
                key = "TotalRuntime"
            else:
                key = phase.replace(" ", "")
            
            runtimes[key] = row['runtime_minutes']
        return runtimes
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def collect_experiment_results(base_path: str, output_csv_path: str = None) -> pd.DataFrame:
    experiment_dirs = list(Path(base_path).glob("experiment_*"))
    print(f"Found {len(experiment_dirs)} experiment directories")

    def safe_read_file(path, parser_fn, default=None):
        try:
            if path.exists():
                return parser_fn(path)
            return default
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return default

    data_records = []
    skipped_count = 0
    termination_count = 0
    missing_config_count = 0
    missing_metrics_count = 0

    for exp_dir in experiment_dirs:
        exp_dir = Path(exp_dir)

        # Check for termination log
        if (exp_dir / "termination_log.csv").exists():
            termination_count += 1
            continue

        # Parse new file formats
        config = safe_read_file(exp_dir / "config.json", parse_config_json)
        metrics = safe_read_file(exp_dir / "trained_model" / "metrics.csv", parse_metrics_csv)
        reid_rate = safe_read_file(exp_dir / "re_identification_results" / "summary_fuzzy_and_greedy.csv", parse_summary_csv)
        reid_rate_fuzzy = safe_read_file(exp_dir / "re_identification_results" / "summary_fuzzy.csv", parse_summary_csv)
        reid_rate_greedy = safe_read_file(exp_dir / "re_identification_results" / "summary_greedy.csv", parse_summary_csv)
        runtime = safe_read_file(exp_dir / "dea_runtime_log.csv", parse_runtime_csv)
        best_result = safe_read_file(
            exp_dir / "hyperparameteroptimization" / "best_result.json",
            parse_best_result_json
        )

        if not config:
            missing_config_count += 1
            print(f"Missing config for: {exp_dir.name}")
            continue

        if not metrics:
            missing_metrics_count += 1
            print(f"Missing metrics for: {exp_dir.name}")
            continue

        record = {
            "ExperimentFolder": exp_dir.name,
            "Encoding": config["ENC_CONFIG"]["AliceAlgo"],
            "Dataset": os.path.basename(config["GLOBAL_CONFIG"]["Data"]),
            "DropFrom": config["GLOBAL_CONFIG"]["DropFrom"],
            "Overlap": config["GLOBAL_CONFIG"]["Overlap"],
            "TrainedPrecision": metrics.get("avg_precision"),
            "TrainedRecall": metrics.get("avg_recall"),
            "TrainedF1": metrics.get("avg_f1"),
            "TrainedDice": metrics.get("avg_dice"),
            "ReidentificationRate": reid_rate,
            "ReidentificationRateFuzzy": reid_rate_fuzzy,
            "ReidentificationRateGreedy": reid_rate_greedy,
        }

        # Include flattened best config
        if best_result:
            record.update({
                "HypOpOutputDim": best_result.get("output_dim"),
                "HypOpNumLayers": best_result.get("num_layers"),
                "HypOpHiddenSize": best_result.get("hidden_layer_size"),
                "HypOpDropout": best_result.get("dropout_rate"),
                "HypOpActivation": best_result.get("activation_fn"),
                "HypOpOptimizer": best_result.get("optimizer"),        # full dict or string
                "HypOpLossFn": best_result.get("loss_fn"),
                "HypOpThreshold": best_result.get("threshold"),
                "HypOpLRScheduler": best_result.get("lr_scheduler"),   # full dict or string
                "HypOpBatchSize": best_result.get("batch_size"),
                "HypOpValLoss": best_result.get("total_val_loss"),
                "HypOpEpochs": best_result.get("epochs"),
                "HypOpF1": best_result.get("average_f1"),
                "HypOpPrecision": best_result.get("average_precision"),
                "HypOpRecall": best_result.get("average_recall"),
                "HypOpDice": best_result.get("average_dice"),
                "LenTrain": best_result.get("len_train"),
                "LenVal": best_result.get("len_val"),
            })

        # Merge runtime fields into the record
        if runtime:
            record.update({
                "GraphMatchingAttackTime": runtime.get("GraphMatchingAttack"),
                "HyperparameterOptimizationTime": runtime.get("HyperparameterOptimization"),
                "ModelTrainingTime": runtime.get("ModelTraining"),
                "ApplicationtoEncodedDataTime": runtime.get("ApplicationtoEncodedData"),
                "RefinementandReconstructionTime": runtime.get("RefinementandReconstruction"),
                "TotalRuntime": runtime.get("TotalRuntime"),
            })

        data_records.append(record)

    print(f"\nExtraction Summary:")
    print(f"Total experiments found: {len(experiment_dirs)}")
    print(f"Experiments with termination logs: {termination_count}")
    print(f"Experiments missing config: {missing_config_count}")
    print(f"Experiments missing metrics: {missing_metrics_count}")
    print(f"Successfully extracted: {len(data_records)}")

    df = pd.DataFrame(data_records)
    if output_csv_path:
        df.to_csv(output_csv_path, index=False)
    return df

# Update the call to use the correct path
collect_experiment_results("experiment_results", "formatted_results.csv")
