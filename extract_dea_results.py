import os
import json
import re
import pandas as pd
from pathlib import Path
import ast
import ast

def parse_best_result_csv(path):
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        row = df.iloc[0].to_dict()

        for key in ["optimizer", "lr_scheduler"]:
            if key in row and isinstance(row[key], str):
                try:
                    row[key] = ast.literal_eval(row[key])
                except:
                    pass
        return row
    except:
        return None



def parse_config_txt(path):
    config = {}
    with open(path, 'r') as f:
        raw = f.read()
    for block in raw.split("# === ")[1:]:
        name, content = block.split(" ===\n", 1)
        config[name.strip()] = json.loads(content)
    return config

def parse_metrics_txt(path):
    metrics = {}
    with open(path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':')
                metrics[key.strip()] = float(value)
    return metrics

def parse_summary_txt(path):
    with open(path, 'r') as f:
        text = f.read()
    match = re.search(r"(Reidentification Rate|Combined Reidentification Rate):\s*([\d.]+)%", text)
    return float(match.group(2)) / 100 if match else None

def parse_runtime_txt(path):
    with open(path, 'r') as f:
        text = f.read()

    # Dictionary to store all extracted runtimes
    runtimes = {}

    # Regex pattern for all steps
    pattern = r"([\w\s]+):\s*([\d.]+)m"

    for step, minutes in re.findall(pattern, text):
        key = step.strip().replace(" ", "")  # e.g., "GraphMatchingAttack"
        runtimes[key] = float(minutes)

    return runtimes


def collect_experiment_results(base_path: str, output_csv_path: str = None) -> pd.DataFrame:
    experiment_dirs = list(Path(base_path).glob("experiment_*"))
    print(f"Found {len(experiment_dirs)} experiment directories")

    def safe_read_file(path, parser_fn, default=None):
        try:
            return parser_fn(path)
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
        if (exp_dir / "termination_log.txt").exists():
            termination_count += 1
            continue

        config = safe_read_file(exp_dir / "config.txt", parse_config_txt)
        metrics = safe_read_file(exp_dir / "trained_model" / "metrics.txt", parse_metrics_txt)
        reid_rate = safe_read_file(exp_dir / "re_identification_results" / "summary_fuzzy_and_greedy.txt", parse_summary_txt)
        reid_rate_fuzzy = safe_read_file(exp_dir / "re_identification_results" / "summary_fuzzy.txt", parse_summary_txt)
        reid_rate_greedy = safe_read_file(exp_dir / "re_identification_results" / "summary_greedy.txt", parse_summary_txt)
        runtime = safe_read_file(exp_dir / "dea_runtime_log.txt", parse_runtime_txt)
        best_result = safe_read_file(
            exp_dir / "hyperparameteroptimization" / "Best_Result.csv",
            parse_best_result_csv
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
            "TrainedPrecision": metrics.get("Average Precision"),
            "TrainedRecall": metrics.get("Average Recall"),
            "TrainedF1": metrics.get("Average F1 Score"),
            "TrainedDice": metrics.get("Average Dice Similarity"),
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
                "HypOpBatchSize": best_result.get("batch_size"),
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

collect_experiment_results("experiment_results/experiment_results_filtered", "formatted_results.csv")
