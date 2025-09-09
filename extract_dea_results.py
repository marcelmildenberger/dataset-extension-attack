import os
import json
import pandas as pd
from pathlib import Path

def read_json_file(file_path):
    """Read and parse JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def read_csv_file(file_path):
    """Read and parse CSV file"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def extract_reidentification_rate(summary_df):
    """Extract reidentification rate from summary CSV"""
    if summary_df is None:
        return None
    
    for _, row in summary_df.iterrows():
        if 'reidentification rate' in str(row.get('metric', '')).lower():
            value = row.get('value', '')
            if isinstance(value, str) and '%' in value:
                try:
                    rate_str = value.replace('%', '').strip()
                    return float(rate_str) / 100
                except:
                    pass
    return None

def extract_metrics(metrics_df):
    """Extract metrics from trained model CSV"""
    if metrics_df is None:
        return {}
    
    metrics = {}
    for _, row in metrics_df.iterrows():
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

def extract_runtime(runtime_df):
    """Extract runtime information from DEA runtime log"""
    if runtime_df is None:
        return {}
    
    runtimes = {}
    for _, row in runtime_df.iterrows():
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

def extract_experiment_data(exp_dir):
    """Extract all data from a single experiment directory"""
    exp_path = Path(exp_dir)
    
    # Check for termination log - skip if present
    if (exp_path / "termination_log.csv").exists():
        return None
    
    # Read configuration
    config = read_json_file(exp_path / "config.json")
    if not config:
        return None
    
    # Read metrics
    metrics_df = read_csv_file(exp_path / "trained_model" / "metrics.csv")
    metrics = extract_metrics(metrics_df)
    if not metrics:
        return None
    
    # Read reidentification results
    reid_summary_df = read_csv_file(exp_path / "re_identification_results" / "summary_fuzzy_and_greedy.csv")
    reid_rate = extract_reidentification_rate(reid_summary_df)
    
    reid_fuzzy_df = read_csv_file(exp_path / "re_identification_results" / "summary_fuzzy.csv")
    reid_rate_fuzzy = extract_reidentification_rate(reid_fuzzy_df)
    
    reid_greedy_df = read_csv_file(exp_path / "re_identification_results" / "summary_greedy.csv")
    reid_rate_greedy = extract_reidentification_rate(reid_greedy_df)
    
    # Read runtime data
    runtime_df = read_csv_file(exp_path / "dea_runtime_log.csv")
    runtime = extract_runtime(runtime_df)
    
    # Read hyperparameter optimization results
    best_result = read_json_file(exp_path / "hyperparameteroptimization" / "best_result.json")
    
    # Create record
    record = {
        "ExperimentFolder": exp_path.name,
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
    
    # Add hyperparameter optimization results
    if best_result:
        record.update({
            "HypOpOutputDim": best_result.get("output_dim"),
            "HypOpNumLayers": best_result.get("num_layers"),
            "HypOpHiddenSize": best_result.get("hidden_layer_size"),
            "HypOpDropout": best_result.get("dropout_rate"),
            "HypOpActivation": best_result.get("activation_fn"),
            "HypOpOptimizer": best_result.get("optimizer"),
            "HypOpLossFn": best_result.get("loss_fn"),
            "HypOpThreshold": best_result.get("threshold"),
            "HypOpLRScheduler": best_result.get("lr_scheduler"),
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
    
    # Add runtime information
    if runtime:
        record.update({
            "GraphMatchingAttackTime": runtime.get("GraphMatchingAttack"),
            "HyperparameterOptimizationTime": runtime.get("HyperparameterOptimization"),
            "ModelTrainingTime": runtime.get("ModelTraining"),
            "ApplicationtoEncodedDataTime": runtime.get("ApplicationtoEncodedData"),
            "RefinementandReconstructionTime": runtime.get("RefinementandReconstruction"),
            "TotalRuntime": runtime.get("TotalRuntime"),
        })
    
    return record

def main():
    """Main function to extract all experiment results"""
    base_path = "experiment_results"
    output_file = "formatted_results.csv"
    
    # Find all experiment directories
    experiment_dirs = list(Path(base_path).glob("experiment_*"))
    print(f"Found {len(experiment_dirs)} experiment directories")
    
    # Extract data from each experiment
    data_records = []
    skipped_count = 0
    missing_config_count = 0
    missing_metrics_count = 0
    
    for exp_dir in experiment_dirs:
        print(f"Processing {exp_dir.name}...")
        
        # Check for termination log
        if (exp_dir / "termination_log.csv").exists():
            skipped_count += 1
            print(f"  Skipped (terminated): {exp_dir.name}")
            continue
        
        # Extract data
        record = extract_experiment_data(exp_dir)
        
        if record is None:
            # Check what's missing
            if not (exp_dir / "config.json").exists():
                missing_config_count += 1
                print(f"  Missing config: {exp_dir.name}")
            elif not (exp_dir / "trained_model" / "metrics.csv").exists():
                missing_metrics_count += 1
                print(f"  Missing metrics: {exp_dir.name}")
            else:
                print(f"  Failed to extract: {exp_dir.name}")
            continue
        
        data_records.append(record)
        print(f"  Successfully extracted: {exp_dir.name}")
    
    # Create DataFrame and save
    df = pd.DataFrame(data_records)
    
    print(f"\nExtraction Summary:")
    print(f"Total experiments found: {len(experiment_dirs)}")
    print(f"Experiments with termination logs: {skipped_count}")
    print(f"Experiments missing config: {missing_config_count}")
    print(f"Experiments missing metrics: {missing_metrics_count}")
    print(f"Successfully extracted: {len(data_records)}")
    
    if len(data_records) > 0:
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("No valid experiments found to extract")

if __name__ == "__main__":
    main()