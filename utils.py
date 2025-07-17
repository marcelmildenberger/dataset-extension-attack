# Standard library imports
import csv
from functools import partial
import json
import os
from hashlib import md5
from typing import Sequence

# Third-party imports
import hickle as hkl
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch
from collections import Counter
from dotenv import load_dotenv
from groq import Groq
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
from torch.utils.data import random_split, Subset
from datasets.bloom_filter_dataset import BloomFilterDataset
from datasets.tab_min_hash_dataset import TabMinHashDataset
from datasets.two_step_hash_dataset import TwoStepHashDataset
import seaborn as sns
from string_utils import extract_two_grams, format_birthday, process_file, lowercase_df





load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# List of keys to remove
keys_to_remove = [
    "config", "checkpoint_dir_name", "experiment_tag", "done", "training_iteration",
    "trial_id", "date", "time_this_iter_s", "pid", "time_total_s", "hostname",
    "node_ip", "time_since_restore", "iterations_since_restore", "timestamp"
]

def get_cache_path(data_directory, identifier, alice_enc_hash, name="dataset"):
    os.makedirs(f"{data_directory}/cache", exist_ok=True)
    return os.path.join(data_directory, "cache", f"{name}_{identifier}_{alice_enc_hash}.pkl")

def read_tsv(path: str, skip_header: bool = True, as_dict: bool = False, delim: str = "\t") -> Sequence[Sequence[str]]:
    data = {} if as_dict else []
    uid = []
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter=delim)
        if skip_header:
            header = next(reader)
        else:
            header = next(reader)
        for row in reader:
            if as_dict:
                assert len(row) == 3, "Dict mode only supports rows with two values + uid"
                data[row[0]] = row[1]
            else:
                data.append(row[:-1])
                uid.append(row[-1])
    return data, uid, header


def save_tsv(data, path: str, delim: str = "\t", mode="w", write_header: bool = False, header: list[str]= None):
    with open(path, mode, newline="") as f:
        csvwriter = csv.writer(f, delimiter=delim)
        if(write_header):
            csvwriter.writerow(header)
        csvwriter.writerows(data)


def get_hashes(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG):
     # Compute hashes of configuration to store/load data and thus avoid redundant computations.
    # Using MD5 because Python's native hash() is not stable across processes
    if GLOBAL_CONFIG["DropFrom"] == "Alice":

        eve_enc_hash = md5(
            ("%s-%s-DropAlice" % (str(ENC_CONFIG), GLOBAL_CONFIG["Data"])).encode()).hexdigest()
        alice_enc_hash = md5(
            ("%s-%s-%s-DropAlice" % (str(ENC_CONFIG), GLOBAL_CONFIG["Data"],
                                     GLOBAL_CONFIG["Overlap"])).encode()).hexdigest()

        eve_emb_hash = md5(
            ("%s-%s-%s-DropAlice" % (str(EMB_CONFIG), str(ENC_CONFIG), GLOBAL_CONFIG["Data"])).encode()).hexdigest()

        alice_emb_hash = md5(("%s-%s-%s-%s-DropAlice" % (str(EMB_CONFIG), str(ENC_CONFIG), GLOBAL_CONFIG["Data"],
                                                         GLOBAL_CONFIG["Overlap"])).encode()).hexdigest()
    elif GLOBAL_CONFIG["DropFrom"] == "Eve":

        eve_enc_hash = md5(
            ("%s-%s-%s-DropEve" % (str(ENC_CONFIG), GLOBAL_CONFIG["Data"],
                                   GLOBAL_CONFIG["Overlap"])).encode()).hexdigest()

        alice_enc_hash = md5(("%s-%s-DropEve" % (str(ENC_CONFIG), GLOBAL_CONFIG["Data"])).encode()).hexdigest()

        eve_emb_hash = md5(("%s-%s-%s-%s-DropEve" % (str(EMB_CONFIG), str(ENC_CONFIG), GLOBAL_CONFIG["Data"],
                                                     GLOBAL_CONFIG["Overlap"])).encode()).hexdigest()

        alice_emb_hash = md5(("%s-%s-%s-DropEve" % (str(EMB_CONFIG), str(ENC_CONFIG),
                                                    GLOBAL_CONFIG["Data"])).encode()).hexdigest()
    else:
        eve_enc_hash = md5(
            ("%s-%s-%s-DropBoth" % (str(ENC_CONFIG), GLOBAL_CONFIG["Data"],
                                    GLOBAL_CONFIG["Overlap"])).encode()).hexdigest()

        alice_enc_hash = md5(
            ("%s-%s-%s-DropBoth" % (str(ENC_CONFIG), GLOBAL_CONFIG["Data"],
                                    GLOBAL_CONFIG["Overlap"])).encode()).hexdigest()

        eve_emb_hash = md5(("%s-%s-%s-%s-DropBoth" % (str(EMB_CONFIG), str(ENC_CONFIG), GLOBAL_CONFIG["Data"],
                                                      GLOBAL_CONFIG["Overlap"])).encode()).hexdigest()

        alice_emb_hash = md5(("%s-%s-%s-%s-DropBoth" % (str(EMB_CONFIG), str(ENC_CONFIG), GLOBAL_CONFIG["Data"],
                                                        GLOBAL_CONFIG["Overlap"])).encode()).hexdigest()

    return eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash

def precision_recall_f1(y_true, y_pred):
    true_set, pred_set = set(y_true), set(y_pred)

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return precision, recall, f1


def run_epoch(model, dataloader, criterion, optimizer, device, is_training, verbose, scheduler=None):
    model.train() if is_training else model.eval()
    running_loss = 0.0

    data_iter = tqdm(dataloader, desc="Training" if is_training else "Validation") if verbose else dataloader

    with torch.set_grad_enabled(is_training):
        for data, labels, _ in data_iter:
            data, labels = data.to(device), labels.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()
                if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()

            running_loss += loss.item() * labels.size(0)

    return running_loss / len(dataloader.dataset)



def map_probabilities_to_two_grams(two_gram_dict, probabilities):
    return [
        {two_gram_dict[j]: prob.item() for j, prob in enumerate(sample)}
        for sample in probabilities
    ]



def filter_high_scoring_two_grams(two_gram_scores, threshold, max_grams=33):
    filtered = []
    for score_dict in two_gram_scores:
        # Filter by threshold
        filtered_grams = [(gram, score) for gram, score in score_dict.items() if score > threshold]
        # Sort by score descending and keep only top `max_grams`
        top_grams = sorted(filtered_grams, key=lambda x: x[1], reverse=True)[:max_grams]
        # Extract just the 2-grams
        filtered.append([gram for gram, _ in top_grams])
    return filtered



def calculate_performance_metrics(actual_batch, predicted_batch):
    total_precision = total_recall = total_f1 = total_dice = 0.0

    for actual, predicted in zip(actual_batch, predicted_batch):
        precision, recall, f1 = precision_recall_f1(actual, predicted)
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_dice += dice_coefficient(actual, predicted)

    return total_dice, total_precision, total_recall, total_f1


def decode_labels_to_two_grams(two_gram_dict, label_batch):
    return [
        [two_gram_dict[i] for i, val in enumerate(label_tensor) if val == 1]
        for label_tensor in label_batch
    ]


def reconstruct_identities_with_llm(result, columns=["GivenName", "Surname", "Birthday"]):
    client = Groq(api_key=groq_api_key)
    all_results = []

    given_name_col, surname_col, birthday_col = columns

    def format_input(batch):
        return "\n".join(
            f'"{entry["uid"]}": ["{", ".join(sorted(entry["predicted_two_grams"]))}"]'
            for entry in batch
        )

    batches = [result[i:i + 15] for i in range(0, len(result), 15)]

    for batch in batches:
        prompt = (
            f"You are an attacker attempting to reconstruct the {given_name_col}, {surname_col}, "
            f"and {birthday_col} of multiple individuals based on 2-grams extracted from a dataset extension attack.\n\n"
            "Each individual is represented by a UID and a list of predicted 2-grams. For each individual, infer:\n"
            f"- {given_name_col}\n- {surname_col}\n- {birthday_col} (in M/D/YYYY format, without leading zeros)\n\n"
            "Only return valid JSON in the following format:\n"
            "[\n"
            "  {\n"
            "    \"uid\": \"29995\",\n"
            f"    \"{given_name_col}\": \"Leslie\",\n"
            f"    \"{surname_col}\": \"Smith\",\n"
            f"    \"{birthday_col}\": \"12/22/1974\"\n"
            "  },\n"
            "  ...\n"
            "]\n\n"
            "Here is the input:\n"
            "{\n" + format_input(batch) + "\n}"
        )



        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="gemma2-9b-it",
                stream=False,
            )

            response_text = response.choices[0].message.content
            json_str = response_text[response_text.find("["):response_text.rfind("]") + 1]
            all_results.extend(json.loads(json_str))

        except Exception as e:
            print("Failed to parse JSON:", e)
            print("Raw response:\n", response_text)
            continue

    return all_results


def print_and_save_result(label, result, save_to):
    print(f"\n {label}")
    print("-" * 40)

    config = resolve_config(result.config)
    metrics = result.metrics

    print(f"Config: {config}")
    print(f"Average Dice: {metrics.get('average_dice'):.4f}")
    print(f"Average Precision: {metrics.get('average_precision'):.4f}")
    print(f"Average Recall: {metrics.get('average_recall'):.4f}")
    print(f"Average F1: {metrics.get('average_f1'):.4f}")

    result_record = {**config, **metrics}
    clean_result_dict(result_record)

    pd.DataFrame([result_record]).to_csv(f"{save_to}/{label}.csv", index=False)


def clean_result_dict(result_dict):
    for key in keys_to_remove:
        result_dict.pop(key, None)
    return result_dict

def resolve_config(config):
    resolved = {}

    for key, value in config.items():
        if isinstance(value, dict):
            # Recursively resolve nested dictionaries
            resolved[key] = resolve_config(value)
        elif isinstance(value, (int, float, str, Subset)):
            # Use the value as-is if it's a basic type or Subset
            resolved[key] = value
        else:
            # Assume it's a Ray Tune search space object and sample a concrete value
            resolved[key] = value.sample()

    return resolved


def metrics_per_entry(actual, predicted):
    actual_set, predicted_set = set(actual), set(predicted)
    precision, recall, f1 = precision_recall_f1(actual_set, predicted_set)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "actual_len": len(actual_set),
        "predicted_len": len(predicted_set),
        "dice": dice_coefficient(actual_set, predicted_set),
        "jaccard": jaccard_similarity(actual_set, predicted_set),
    }


def dice_coefficient(set1: set, set2: set) -> float:
    set1, set2 = set(set1), set(set2)
    if not set1 and not set2:
        return 1.0  # both empty sets → full similarity
    intersection = len(set1 & set2)
    return round(2 * intersection / (len(set1) + len(set2)), 4)


def jaccard_similarity(set1, set2):
    set1, set2 = set(set1), set(set2)
    if not set1 and not set2:
        return 1.0  # both empty sets → full similarity
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


def find_most_likely_birthday(entry, all_birthday_records, similarity_metric='jaccard'):
    similarity_func = {
        'jaccard': jaccard_similarity,
        'dice': dice_coefficient
    }.get(similarity_metric)

    if similarity_func is None:
        raise ValueError("similarity_metric must be 'jaccard' or 'dice'")

    best_birthday, best_score, best_grams = max(
        (
            (birthday, similarity_func(entry['predicted_two_grams'], grams), grams)
            for birthday, grams in all_birthday_records
        ),
        key=lambda x: x[1],
        default=(None, -1, [])
    )

    filtered_grams = [
        gram for gram in entry['predicted_two_grams'] if gram not in best_grams
    ]

    entry["predicted_two_grams"] = filtered_grams

    return best_birthday, best_score

def find_most_likely_given_name(entry, all_givenname_records, similarity_metric='jaccard'):
    similarity_func = {
        'jaccard': jaccard_similarity,
        'dice': dice_coefficient
    }.get(similarity_metric)

    if similarity_func is None:
        raise ValueError("similarity_metric must be 'jaccard' or 'dice'")

    best_name, best_score, best_name_grams = max(
        (
            (name, similarity_func(entry['predicted_two_grams'], grams), grams)
            for name, grams in all_givenname_records
        ),
        key=lambda x: x[1],
        default=(None, -1, [])
    )

    filtered_grams = [
        gram for gram in entry['predicted_two_grams'] if gram not in best_name_grams
    ]

    entry["predicted_two_grams"] = filtered_grams

    return best_name, best_score, filtered_grams

def find_most_likely_surname(entry, all_surname_records, similarity_metric='jaccard'):
    similarity_func = {
        'jaccard': jaccard_similarity,
        'dice': dice_coefficient
    }.get(similarity_metric)

    if similarity_func is None:
        raise ValueError("similarity_metric must be 'jaccard' or 'dice'")

    best_name, best_score, best_grams = max(
        (
            (name, similarity_func(entry['predicted_two_grams'], grams), grams)
            for name, grams in all_surname_records
        ),
        key=lambda x: x[1],
        default=(None, -1, set())
    )

    filtered_grams = [
        gram for gram in entry['predicted_two_grams'] if gram not in best_grams
    ]

    entry["predicted_two_grams"] = filtered_grams

    return best_name, best_score, filtered_grams



def greedy_reconstruction(result):
    reconstructed_results = []

    for entry in result:
        uid = entry["uid"]
        two_grams = entry["predicted_two_grams"]


        # Build directed graph from 2-grams
        G = nx.DiGraph()
        G.add_edges_from((gram[0], gram[1]) for gram in two_grams)

        if nx.is_directed_acyclic_graph(G):
            path = nx.dag_longest_path(G)
            reconstructed = path[0] + ''.join(path[1:]) if path else ""
        else:
            # DFS to find longest possible sequence
            def dfs(node, visited_edges, current_string):
                nonlocal longest_sequence
                if len(current_string) > len(longest_sequence):
                    longest_sequence = current_string

                for neighbor in G.successors(node):
                    edge = (node, neighbor)
                    if edge not in visited_edges:
                        visited_edges.add(edge)
                        dfs(neighbor, visited_edges, current_string + neighbor)
                        visited_edges.remove(edge)

            longest_sequence = ""
            for gram in two_grams:
                dfs(gram[1], {(gram[0], gram[1])}, gram[0] + gram[1])

            reconstructed = longest_sequence

        reconstructed_results.append({
            "uid": uid,
            "identifier": reconstructed
        })

    return reconstructed_results

def load_givenname_and_surname_records(min_count=10, use_filtered=True):
    file_path = (
        'data/names/surname/Filtered_Names_2010Census.csv'
        if use_filtered else 'data/names/surname/Names_2010Census.csv'
    )

    all_surname_records = []
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row['count']) >= min_count:
                    name = row['name'].lower()
                    grams = extract_two_grams(name)
                    all_surname_records.append((name, set(grams)))

    # Load and preprocess all given names into 2-grams
    with open('data/names/givenname/unique_names.txt', 'r', encoding='utf-8') as f:
        all_givenname_records = [
            (name.strip(), extract_two_grams(name.strip()))
            for name in f if name.strip()
        ]
    return all_givenname_records, all_surname_records

def load_birthday_2gram_records():
    date_range = pd.date_range(start="1900-01-01", end="2004-12-31", freq='D')
    return [
        (d.strftime("%-m/%-d/%Y"), extract_two_grams(d.strftime("%-m/%-d/%Y")))
        for d in date_range
    ]

def create_identifier_column_dynamic(df, components):
    cleaned_cols = [
        df[col].astype(str).str.replace('/', '', regex=False)
        for col in components
    ]
    return pd.Series(map(''.join, zip(*cleaned_cols)), index=df.index).str.lower()


def reidentification_analysis(df_1, df_2, merge_on, len_not_reidentified, method_name=None, save_path=None):
    merged = pd.merge(df_1, df_2, on=merge_on, how='inner', suffixes=('_df1', '_df2'))

    total_reidentified = len(merged)
    total_not_reidentified = len_not_reidentified

    print("Reidentification Analysis:")
    print(f"Technique: {method_name}")
    print(f"Total Reidentified Individuals: {total_reidentified}")
    print(f"Total Not Reidentified Individuals: {total_not_reidentified}")

    if total_not_reidentified > 0:
        reidentification_rate = (total_reidentified / total_not_reidentified) * 100
        print(f"Reidentification Rate: {reidentification_rate:.2f}%")
    else:
        reidentification_rate = None
        print("No not reidentified individuals to analyze.")

    if save_path:
        os.makedirs(save_path, exist_ok=True)

        # Save merged reidentified individuals
        result_csv_path = os.path.join(save_path, f"result_{method_name or 'unknown'}.csv")
        merged.to_csv(result_csv_path, index=False)

        # Save summary
        summary_txt_path = os.path.join(save_path, f"summary_{method_name or 'unknown'}.txt")
        with open(summary_txt_path, "w") as f:
            f.write(f"Reidentification Method: {method_name or 'Unknown'}\n")
            f.write(f"Total Reidentified Individuals: {total_reidentified}\n")
            f.write(f"Total Not Reidentified Individuals: {total_not_reidentified}\n")
            if reidentification_rate is not None:
                f.write(f"Reidentification Rate: {reidentification_rate:.2f}%\n")
            else:
                f.write("No not reidentified individuals to analyze.\n")

    return merged


# Convert seconds to minutes
def to_minutes(seconds):
    return round(seconds / 60, 2)


def save_dea_runtime_log(
    elapsed_gma,
    elapsed_hyperparameter_optimization,
    elapsed_model_training,
    elapsed_application_to_encoded_data,
    elapsed_refinement_and_reconstruction,
    elapsed_total,
    output_dir="dea_runtime_logs"
):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "dea_runtime_log.txt")

    runtimes = {
        "Graph Matching Attack": elapsed_gma,
        "Hyperparameter Optimization": elapsed_hyperparameter_optimization,
        "Model Training": elapsed_model_training,
        "Application to Encoded Data": elapsed_application_to_encoded_data,
        "Refinement and Reconstruction": elapsed_refinement_and_reconstruction,
        "Total Runtime": elapsed_total
    }

    with open(output_file, "w") as f:
        f.write("DEA Runtime (in minutes):\n")
        for label, seconds in runtimes.items():
            f.write(f"{label}: {to_minutes(seconds)}m\n")


def load_dataframe(path):
    data = hkl.load(path)
    return pd.DataFrame(data[1:], columns=data[0])

def fake_name_analysis():
    dataset_sizes = [1000, 2000, 5000, 10000, 20000, 50000]
    precisions = [0.2162, 0.2131, 0.2144, 0.2151, 0.2153, 0.2151]
    recalls = [0.2476, 0.2452, 0.2470, 0.2467, 0.2473, 0.2463]
    f1_scores = [0.2300, 0.2271, 0.2287, 0.2289, 0.2293, 0.2288]

    plt.plot(dataset_sizes, precisions, label='Precision', marker='o')
    plt.plot(dataset_sizes, recalls, label='Recall', marker='s')
    plt.plot(dataset_sizes, f1_scores, label='F1 Score', marker='^')
    plt.xlabel('Dataset Size')
    plt.ylabel('Metric Value')
    plt.title('Baseline Guessing Performance on Fakename Datasets')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def reconstruct_single_entry(entry, all_givenname_records, all_surname_records, all_birthday_records, reconstruct_birthday):
    """
    Reconstruct a single entry by matching predicted two-grams to the most likely given name, surname, and optionally birthday.
    This function is stateless and safe for parallel execution.
    """
    # Avoid mutating the input entry
    entry_dict = {
        'uid': entry['uid'],
        'actual_two_grams': entry["actual_two_grams"],
        'predicted_two_grams': list(entry['predicted_two_grams'])  # ensure copy
    }

    # Given name
    given_name, _, _ = find_most_likely_given_name(
        entry_dict,
        all_givenname_records=all_givenname_records,
        similarity_metric='dice'
    )

    # Surname
    surname, _, _ = find_most_likely_surname(
        entry_dict,
        all_surname_records=all_surname_records,
        similarity_metric='dice'
    )

    if reconstruct_birthday:
        # Birthday
        birthday, _ = find_most_likely_birthday(
            entry_dict,
            all_birthday_records=all_birthday_records,
            similarity_metric='dice'
        )
        return (given_name, surname, birthday, entry['uid'])

    return (given_name, surname, entry['uid'])


def fuzzy_reconstruction_approach(result, workers, reconstruct_birthday):
    """
    Reconstructs all entries in parallel using joblib.Parallel.
    Loads reference records only once and shares them across workers.
    """

    # Load reference records once (shared, read-only)
    all_birthday_records = load_birthday_2gram_records()
    all_givenname_records, all_surname_records = load_givenname_and_surname_records(min_count=150, use_filtered=True)

    # Use partial to avoid repeatedly passing large reference data
    reconstruct_fn = partial(
        reconstruct_single_entry,
        all_givenname_records=all_givenname_records,
        all_surname_records=all_surname_records,
        all_birthday_records=all_birthday_records,
        reconstruct_birthday=reconstruct_birthday
    )

    # Dynamically choose batch_size for optimal parallel efficiency
    if len(result) > 5000:
        batch_size = 500
    elif len(result) > 2000:
        batch_size = 200
    elif len(result) > 1000:
        batch_size = 100
    elif len(result) > 200:
        batch_size = 20
    else:
        batch_size = 1

    reconstructed = Parallel(n_jobs=workers, batch_size=batch_size, prefer="processes")(
        delayed(reconstruct_fn)(entry) for entry in result
    )

    return reconstructed

def read_header(tsv_path):
    with open(tsv_path, 'r', encoding='utf-8') as f:
        header_line = f.readline().strip()
        columns = header_line.split('\t')
        return columns


def load_experiment_datasets(
    data_directory, alice_enc_hash, identifier, ENC_CONFIG, DEA_CONFIG, GLOBAL_CONFIG, all_two_grams, splits=("train", "val", "test")
):
    cache_path = get_cache_path(data_directory, identifier, alice_enc_hash)
    # Try to load from cache if all splits are present
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
            if isinstance(cached, dict):
                return {k: cached.get(k) for k in splits}
            else:
                train, val, test = cached
                return {"train": train, "val": val, "test": test}

    # Otherwise, create datasets
    df_reidentified = load_dataframe(f"{data_directory}/available_to_eve/reidentified_individuals_{identifier}.h5")

    df_not_reidentified = load_dataframe(f"{data_directory}/available_to_eve/not_reidentified_individuals_{identifier}.h5")
    df_all = load_dataframe(f"{data_directory}/dev/alice_data_complete_with_encoding_{alice_enc_hash}.h5")
    df_test = df_all[df_all["uid"].isin(df_not_reidentified["uid"])].reset_index(drop=True)

    DatasetClass = None
    algo = ENC_CONFIG["AliceAlgo"]
    if algo == "BloomFilter":
        DatasetClass = BloomFilterDataset
    elif algo == "TabMinHash":
        DatasetClass = TabMinHashDataset
    elif algo == "TwoStepHash":
        DatasetClass = TwoStepHashDataset

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
    data_test = DatasetClass(df_test, **common_args, **dataset_args)
    train_size = int(DEA_CONFIG["TrainSize"] * len(data_labeled))
    val_size = len(data_labeled) - train_size
    data_train, data_val = random_split(data_labeled, [train_size, val_size])
    result = {"train": data_train, "val": data_val, "test": data_test}
    # Save all splits to cache for future use
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)
    # Return only the requested splits/datasets
    return {k: result[k] for k in splits}

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

def get_not_reidentified_df(data_dir: str, identifier: str, alice_enc_hash=None) -> pd.DataFrame:
    """
    Loads and lowercases the DataFrame of not re-identified individuals.
    Args:
        data_dir (str): Path to the data directory.
        identifier (str): Unique identifier for the experiment/data split.
        alice_enc_hash (str, optional): Hash for Alice's encoding. If not provided, must be passed by caller.
    Returns:
        pd.DataFrame: Lowercased DataFrame of not re-identified individuals.
    """
    if alice_enc_hash is None:
        raise ValueError("alice_enc_hash must be provided.")
    df = load_not_reidentified_data(data_dir, alice_enc_hash, identifier)
    return lowercase_df(df)


def create_identifier(df: pd.DataFrame, components):
    """
    Adds an 'identifier' column to the DataFrame using the specified components.
    Args:
        df (pd.DataFrame): Input DataFrame.
        components (list): List of column names to use for identifier creation.
    Returns:
        pd.DataFrame: DataFrame with 'uid' and 'identifier' columns.
    """
    df = df.copy()
    df["identifier"] = create_identifier_column_dynamic(df, components)
    return df[["uid", "identifier"]]


def run_reidentification_once(reconstructed_identities, df_not_reidentified, merge_cols, technique, current_experiment_directory, identifier_components=None):
    """
    Runs the reidentification analysis for a single technique.
    Args:
        reconstructed_identities (list/dict): Reconstructed identities.
        df_not_reidentified (pd.DataFrame): Not re-identified DataFrame.
        merge_cols (list): Columns to merge on.
        technique (str): Name of the technique.
        current_experiment_directory (str): Directory to save results.
        identifier_components (list, optional): Components for identifier creation.
    Returns:
        dict: Result of reidentification_analysis.
    """
    # Convert reconstructed identities to DataFrame and lowercase
    df_reconstructed = lowercase_df(pd.DataFrame(reconstructed_identities, columns=merge_cols))
    # If identifier components are provided, create identifier column in not-reidentified DataFrame
    if identifier_components:
        df_not_reidentified = create_identifier(df_not_reidentified, identifier_components)
    return reidentification_analysis(
        df_reconstructed,
        df_not_reidentified,
        merge_cols,
        len(df_not_reidentified),
        technique,
        save_path=f"{current_experiment_directory}/re_identification_results"
    )


def get_reidentification_techniques(header, include_birthday):
    """
    Returns the dictionary of available reidentification techniques and their configurations.
    Args:
        header (list): List of column names from the dataset.
        include_birthday (bool): Whether to include birthday in the merge columns for fuzzy technique.
    Returns:
        dict: Dictionary mapping technique names to their configs.
    """
    return {
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


def run_selected_reidentification(
    selected,
    techniques,
    results,
    df_not_reid_cached,
    GLOBAL_CONFIG,
    current_experiment_directory,
    data_dir,
    identifier,
    save_dir
):
    """
    Runs the selected reidentification technique(s), handles the 'fuzzy_and_greedy' case, saves results, and prints summaries.
    Args:
        selected (str): Selected technique or 'fuzzy_and_greedy'.
        techniques (dict): Dictionary of available techniques.
        results (list): Model results.
        df_not_reid_cached (pd.DataFrame): Not re-identified DataFrame.
        GLOBAL_CONFIG (dict): Global config.
        current_experiment_directory (str): Directory for experiment results.
        data_dir (str): Data directory.
        identifier (str): Unique identifier for the experiment/data split.
        save_dir (str): Directory to save reidentification results.
    Returns:
        None
    """
    if selected == "fuzzy_and_greedy":
        reidentified = {}
        for name in ("greedy", "fuzzy"):
            info = techniques[name]
            if name == "fuzzy":
                # Fuzzy needs extra arguments
                reconstructed_identities = info["fn"](results, GLOBAL_CONFIG["Workers"], not (GLOBAL_CONFIG["Data"] == "./data/datasets/titanic_full.tsv"))
            else:
                reconstructed_identities = info["fn"](results)
            reidentified[name] = run_reidentification_once(
                reconstructed_identities,
                df_not_reid_cached,
                info["merge_cols"],
                name,
                current_experiment_directory,
                info["identifier_comps"],
            )
        # Combine UIDs from both methods
        uids_greedy = set(reidentified["greedy"]["uid"])
        uids_fuzzy = set(reidentified["fuzzy"]["uid"])
        combined_uids = uids_greedy.union(uids_fuzzy)
        total_reidentified_combined = len(combined_uids)
        len_not_reidentified = len(df_not_reid_cached)
        reidentification_rate_combined = (total_reidentified_combined / len_not_reidentified) * 100
        print("Combined Reidentification (greedy ∪ fuzzy):")
        print(f"Total not re-identified individuals: {len_not_reidentified}")
        print(f"Total Unique Reidentified Individuals: {total_reidentified_combined}")
        print(f"Combined Reidentification Rate: {reidentification_rate_combined:.2f}%")
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
    else:
        if selected not in techniques:
            raise ValueError(f"Unsupported matching technique: {selected}")
        info = techniques[selected]
        if selected == "fuzzy":
            reconstructed_identities = info["fn"](results, GLOBAL_CONFIG["Workers"], not (GLOBAL_CONFIG["Data"] == "./data/datasets/titanic_full.tsv"))
        elif selected == "ai":
            reconstructed_identities = info["fn"](results, info["merge_cols"][:-1])
        else:
            reconstructed_identities = info["fn"](results)
        run_reidentification_once(
            reconstructed_identities,
            df_not_reid_cached,
            info["merge_cols"],
            selected,
            current_experiment_directory,
            info["identifier_comps"],
        )

def log_epoch_metrics(epoch, total_epochs, train_loss, val_loss, tb_writer=None, save_results=False):
    """
    Log and optionally write train/val loss to TensorBoard for a given epoch.
    Args:
        epoch: Current epoch (int)
        total_epochs: Total number of epochs (int)
        train_loss: Training loss (float)
        val_loss: Validation loss (float)
        tb_writer: TensorBoard SummaryWriter (optional)
        save_results: Whether to save results to TensorBoard (bool)
    """
    epoch_str = f"[{epoch + 1}/{total_epochs}]"
    print(f"{epoch_str} Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    if save_results and tb_writer is not None:
        tb_writer.add_scalar("Loss/train", train_loss, epoch + 1)
        tb_writer.add_scalar("Loss/validation", val_loss, epoch + 1)


def plot_loss_curves(train_losses, val_losses, save_path=None, save=False):
    """
    Plot training and validation loss curves. Optionally save to file if save=True.
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the plot (if save=True)
        save: Whether to save the plot to file
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training loss', color='blue')
    plt.plot(val_losses, label='Validation loss', color='red')
    plt.legend()
    plt.title("Training and Validation Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    if save and save_path is not None:
        plt.savefig(save_path)
    plt.close()

def plot_metric_distributions(results_df, trained_model_directory, save=False):
    """
    Plot and optionally save the distribution of precision, recall, F1, dice, and jaccard metrics.
    Args:
        results_df: DataFrame with per-sample metrics
        trained_model_directory: Directory to save the plot if save=True
        save: Whether to save the plot
    """
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
    if save:
        out_path = os.path.join(trained_model_directory, "metric_distributions.png")
        plt.savefig(out_path)
    plt.close()

