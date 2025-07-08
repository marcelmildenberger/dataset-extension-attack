import os
import json
import re
import pandas as pd
from pathlib import Path

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

    def safe_read_file(path, parser_fn, default=None):
        try:
            return parser_fn(path)
        except:
            return default

    data_records = []
    for exp_dir in experiment_dirs:
        exp_dir = Path(exp_dir)
        if (exp_dir / "termination_log.txt").exists():
            continue

        config = safe_read_file(exp_dir / "config.txt", parse_config_txt)
        metrics = safe_read_file(exp_dir / "trained_model" / "metrics.txt", parse_metrics_txt)
        reid_rate = safe_read_file(exp_dir / "re_identification_results" / "summary_fuzzy_and_greedy.txt", parse_summary_txt)
        reid_rate_fuzzy = safe_read_file(exp_dir / "re_identification_results" / "summary_fuzzy.txt", parse_summary_txt)
        reid_rate_greedy = safe_read_file(exp_dir / "re_identification_results" / "summary_greedy.txt", parse_summary_txt)
        runtime = safe_read_file(exp_dir / "dea_runtime_log.txt", parse_runtime_txt)

        if not config or not metrics:
            continue

        record = {
            "ExperimentFolder": exp_dir.name,
            "Encoding": config["ENC_CONFIG"]["AliceAlgo"],
            "Dataset": os.path.basename(config["GLOBAL_CONFIG"]["Data"]),
            "DropFrom": config["GLOBAL_CONFIG"]["DropFrom"],
            "Overlap": config["GLOBAL_CONFIG"]["Overlap"],
            "Precision": metrics.get("Average Precision"),
            "Recall": metrics.get("Average Recall"),
            "F1": metrics.get("Average F1 Score"),
            "Dice": metrics.get("Average Dice Similarity"),
            "ReidentificationRate": reid_rate,
            "ReidentificationRateFuzzy": reid_rate_fuzzy,
            "ReidentificationRateGreedy": reid_rate_greedy,
        }

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


    df = pd.DataFrame(data_records)
    if output_csv_path:
        df.to_csv(output_csv_path, index=False)
    return df

collect_experiment_results("experiment_results", "formatted_results.csv")
