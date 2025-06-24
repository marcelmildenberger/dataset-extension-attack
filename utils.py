# Standard library imports
import csv
import json
import os
from hashlib import md5
from typing import Sequence

# Third-party imports
import hickle as hkl
import networkx as nx
import pandas as pd
import torch
from dotenv import load_dotenv
from groq import Groq
from torch.utils.data import Subset
from tqdm import tqdm


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# List of keys to remove
keys_to_remove = [
    "config", "checkpoint_dir_name", "experiment_tag", "done", "training_iteration",
    "trial_id", "date", "time_this_iter_s", "pid", "time_total_s", "hostname",
    "node_ip", "time_since_restore", "iterations_since_restore", "timestamp"
]

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

def extract_two_grams(input_string, remove_spaces=False):
    chars_to_remove = '"./'
    translation_table = str.maketrans('', '', chars_to_remove)
    cleaned = input_string.translate(translation_table).strip().lower()
    if remove_spaces:
        cleaned = cleaned.replace(' ', '')

    # Generate 2-grams, excluding those containing spaces
    return [cleaned[i:i+2] for i in range(len(cleaned) - 1) if ' ' not in cleaned[i:i+2]]


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



def filter_high_scoring_two_grams(two_gram_scores, threshold):
    return [
        [gram for gram, score in score_dict.items() if score > threshold]
        for score_dict in two_gram_scores
    ]


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


def reconstruct_identities_with_llm(result):
    client = Groq(api_key=groq_api_key)
    all_results = []

    def format_input(batch):
        return "\n".join(
            f'"{entry["uid"]}": ["{", ".join(sorted(entry["predicted_two_grams"]))}"]'
            for entry in batch
        )

    batches = [result[i:i + 15] for i in range(0, len(result), 15)]

    for batch in batches:
        prompt = (
            "You are an attacker attempting to reconstruct the given name, surname, "
            "and date of birth of multiple individuals based on 2-grams extracted from a dataset extension attack.\n\n"
            "Each individual is represented by a UID and a list of predicted 2-grams. For each individual, infer:\n"
            "- GivenName\n- Surname\n- - Birthday (in M/D/YYYY format, without leading zeros)\n\n"
            "Only return valid JSON in the format:\n"
            "[\n"
            "  {\n    \"uid\": \"29995\",\n    \"GivenName\": \"Leslie\",\n"
            "    \"Surname\": \"Smith\",\n    \"Birthday\": \"12/22/1974\"\n  },\n"
            "  ...\n]\n\n"
            "Here is the input:\n{\n" + format_input(batch) + "\n}"
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
    print(f"\n🔍 {label}")
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


def lowercase_df(df):
    return df.apply(lambda col: col.str.lower() if col.dtype == "object" else col)

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


def process_file(filepath):
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            name, _, count = line.strip().split(',')
            grams = extract_two_grams(name)
            records.append((name.lower(), set(grams), int(count)))
    return records


def format_birthday(date_str):
    return f"{date_str[:2]}/{date_str[2:4]}/{date_str[4:]}"

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

    entry["actual_two_grams"] = filtered_grams

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

    entry["actual_two_grams"] = filtered_grams

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

    entry["actual_two_grams"] = filtered_grams

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


def fuzzy_reconstruction_approach(result):
    print("\n🔄 Reconstructing results using fuzzy matching (entry-wise)...")

    all_birthday_records = load_birthday_2gram_records()
    all_givenname_records, all_surname_records = load_givenname_and_surname_records(min_count=10, use_filtered=True)

    reconstructed = []
    for entry in result:

        uid = entry['uid']
        actual_two_grams = entry["actual_two_grams"]
        predicted_two_grams = entry['predicted_two_grams']

        entry_dict = {
            'uid': uid,
            'actual_two_grams': actual_two_grams,
            'predicted_two_grams': predicted_two_grams
        }

        # Step 1: Given name
        given_name, _, _ = find_most_likely_given_name(
            entry_dict,
            all_givenname_records=all_givenname_records,
            similarity_metric='dice'
        )

        #Step 2: Surname
        surname, _, _ = find_most_likely_surname(
        entry=entry_dict,
        all_surname_records=all_surname_records,
        similarity_metric='dice'
        )

        # Step 3: Birthday
        birthday, _ = find_most_likely_birthday(
        entry_dict,
        all_birthday_records=all_birthday_records,
        similarity_metric='dice'
        )

        # Collect reconstructed entry
        reconstructed.append((given_name, surname, birthday, uid))

    return reconstructed


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

    print("\n🔍 Reidentification Analysis:")
    print(f"Total Reidentified Individuals: {total_reidentified}")
    print(f"Total Not Reidentified Individuals: {total_not_reidentified}")

    if total_not_reidentified > 0:
        reidentification_rate = (total_reidentified / total_not_reidentified) * 100
        print(f"Reidentification Rate: {reidentification_rate:.2f}%")
    else:
        reidentification_rate = None
        print("⚠️ No not reidentified individuals to analyze.")

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

