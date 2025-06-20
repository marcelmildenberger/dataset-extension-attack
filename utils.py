import csv
import json
from typing import List, Sequence, Set
from hashlib import md5
from collections import defaultdict
from groq import Groq
import os
import pandas as pd
import torch
from tqdm import tqdm
from dotenv import load_dotenv
import networkx as nx

from torch.utils.data import Subset
import glob

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
    input_string_preprocessed = input_string.replace('"', '').replace('.', '').replace('/', '').strip()
    if(remove_spaces):
        input_string_preprocessed = input_string_preprocessed.replace(' ', '')
    input_string_lower = input_string_preprocessed.lower()  # Normalize to lowercase for consistency
    return [input_string_lower[i:i+2] for i in range(len(input_string_lower)-1) if ' ' not in input_string_lower[i:i+2]]

def precision_recall_f1(y_true, y_pred):
    true_set = set(y_true)
    pred_set = set(y_pred)

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def run_epoch(model, dataloader, criterion, optimizer, device, is_training, verbose, scheduler):
    running_loss = 0.0
    with torch.set_grad_enabled(is_training):
        for data, labels, _ in tqdm(dataloader,
                                  desc="Training" if is_training else "Validation") if verbose else dataloader:
            data, labels = data.to(device), labels.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()
                if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()

            running_loss += loss.item() * labels.size(0)

    return running_loss / len(dataloader.dataset)


def convert_to_two_gram_scores(two_gram_dict, probabilities):
    return [
        {two_gram_dict[j]: score.item() for j, score in enumerate(probabilities[i])}
        for i in range(probabilities.size(0))
    ]


def filter_two_grams(two_gram_scores, threshold):
    return [
        [two_gram for two_gram, score in two_gram_score.items() if score > threshold]
        for two_gram_score in two_gram_scores
    ]

def calculate_performance_metrics(actual_two_grams_batch, filtered_two_grams):
    sum_precision = 0
    sum_recall = 0
    sum_f1 = 0
    sum_dice = 0

    for actual_two_grams, filtered_two_grams in zip(actual_two_grams_batch, filtered_two_grams):
        precision, recall, f1 = precision_recall_f1(actual_two_grams, filtered_two_grams)
        sum_dice += dice_coefficient(actual_two_grams, filtered_two_grams)
        sum_precision += precision
        sum_recall += recall
        sum_f1 += f1
    return sum_dice, sum_precision, sum_recall, sum_f1

def label_tensors_to_two_grams(two_gram_dict, labels):
    batch_selected_2grams = []
    for label_tensor in labels:  # Loop through each batch element (1D tensor)
        selected_2grams = [two_gram_dict[i] for i, val in enumerate(label_tensor) if val == 1]
        batch_selected_2grams.append(selected_2grams)
    return batch_selected_2grams

def reconstruct_using_ai(result):
    client = Groq(api_key=groq_api_key)

    batches = [result[i:i + 15] for i in range(0, len(result), 15)]
    all_llm_results = []
    for batch in batches:
        prompt = f"""
        You are an attacker attempting to reconstruct the given name, surname, and date of birth of multiple individuals based on 2-grams extracted from a dataset extension attack.

        Each individual is represented by a unique UID and a list of predicted 2-grams. For each individual, infer the most likely real-world values for their:
        - GivenName
        - Surname
        - Birthday (in MM/DD/YYYY format)

        The data is synthetic and anonymized. Return **only** valid JSON in the following format: a list of dictionaries, each containing the UID and the inferred values.

        [
        {{
                "uid": "29995",
                "GivenName": "Leslie",
                "Surname": "Smith",
                "Birthday": "12/22/1974"
            }},
            {{
                "uid": "39734",
                "GivenName": "John",
                "Surname": "Simons",
                "Birthday": "01/25/2000"
            }}
        ]

        Here is the input:
        {{
        {chr(10).join([f'"{entry["uid"]}": ["{", ".join(sorted(entry["filtered_two_grams"]))}"]' for entry in batch])}
        }}
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gemma2-9b-it",
            stream=False,
        )

        response_text = chat_completion.choices[0].message.content

        try:
            # Find the first square bracket and attempt to load JSON from there
            start_index = response_text.find('[')
            end_index = response_text.rfind(']') + 1
            json_str = response_text[start_index:end_index]

            data = json.loads(json_str)
            all_llm_results.extend(data)

        except Exception as e:
            print(f"Failed to parse JSON:", e)
            print("Raw response:\n", response_text)
            continue  # Skip this batch and proceed
    return all_llm_results

def print_and_save_result(label, result, save_to):
    print(f"\n🔍 {label}")
    print("-" * 40)
    cleaned_config = resolve_config(result.config)
    print(f"Config: {cleaned_config}")
    print(f"Average Dice: {result.metrics.get('average_dice'):.4f}")
    print(f"Average Precision: {result.metrics.get('average_precision'):.4f}")
    print(f"Average Recall: {result.metrics.get('average_recall'):.4f}")
    print(f"Average F1: {result.metrics.get('average_f1'):.4f}")
    result_dict = {**cleaned_config, **result.metrics}
    clean_result_dict(result_dict)
    # Convert to a DataFrame and save
    df = pd.DataFrame([result_dict])
    df.to_csv(f"{save_to}/{label}.csv", index=False)

def clean_result_dict(result_dict):
    for key in keys_to_remove:
        result_dict.pop(key, None)
    return result_dict

def resolve_config(config):
    resolved = {}
    for k, v in config.items():
        # If the value is a dictionary, recurse and apply resolve_config
        if isinstance(v, dict):
            resolved[k] = resolve_config(v)
        # If the value is a Ray search sample object (e.g., Float, Categorical)
        elif not isinstance(v, (int, float, str, Subset)):
            resolved[k] = v.sample()  # Get the concrete value from the sample
        else:
            resolved[k] = v  # Leave it as-is if it's not a sample object or Subset
    return resolved

def two_gram_overlap(row):
    actual = set(row['actual_two_grams'])
    predicted = set(row['filtered_two_grams'])
    intersection = actual & predicted
    return {
        "uid": row["uid"],
        "precision": len(intersection) / len(predicted) if predicted else 0,
        "recall": len(intersection) / len(actual) if actual else 0,
        "f1": 2 * len(intersection) / (len(actual) + len(predicted)) if actual and predicted else 0,
        "actual_len": len(actual),
        "predicted_len": len(predicted)
    }

def lowercase_df(df):
    return df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

def get_2grams(name):
    return {name[i:i+2] for i in range(len(name) - 1)}

def dice_coefficient(set1: set, set2: set) -> float:
    if isinstance(set1, list):
        set1 = set(set1)
    if isinstance(set2, list):
        set2 = set(set2)
    if not set1 and not set2:
        return 1.0  # both empty sets → full similarity
    intersection = len(set1 & set2)
    return round((2 * intersection) / (len(set1) + len(set2)),4)

def jaccard_similarity(set1, set2):
    if isinstance(set1, list):
        set1 = set(set1)
    if isinstance(set2, list):
        set2 = set(set2)
    if not set1 and not set2:
        return 1.0  # both empty sets → full similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def process_file(filepath):
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            name, _, count = line.strip().split(',')
            count = int(count)
            grams = get_2grams(name)
            records.append((name.lower(), {g.lower() for g in grams}, count))
    return records

def format_birthday(date_str):
    return f"{date_str[:2]}/{date_str[2:4]}/{date_str[4:]}"

def find_most_likely_birthday(predicted_2grams_list, similarity_metric='jaccard'):
    # Choose similarity function
    if similarity_metric == 'jaccard':
        similarity_func = jaccard_similarity
    elif similarity_metric == 'dice':
        similarity_func = dice_coefficient
    else:
        raise ValueError("similarity_metric must be 'jaccard' or 'dice'")

    date_range = pd.date_range(start="1900-01-01", end="2004-12-31", freq='D')
    candidate_birthdays = [d.strftime("%m%d%Y") for d in date_range]

    candidate_2gram_list = [(bd, get_2grams(bd)) for bd in candidate_birthdays]

    best_matches = []
    for entry in predicted_2grams_list:
        best_birthday = None
        best_score = -1
        # Calculate similarity for each candidate birthday
        for (birthday, birthday_grams) in candidate_2gram_list:
            score = similarity_func(entry['filtered_two_grams'], birthday_grams)
            if score > best_score:
                best_score = score
                best_birthday = birthday
        # Sort candidates by score and take the best one
        best_matches.append((format_birthday(best_birthday), best_score, entry['uid']))
    return best_matches

def find_most_likely_given_name(predicted_2grams_list, unique_names_file='data/names/givenname/unique_names.txt', similarity_metric='jaccard'):
    # Choose similarity function
    if similarity_metric == 'jaccard':
        similarity_func = jaccard_similarity
    elif similarity_metric == 'dice':
        similarity_func = dice_coefficient
    else:
        raise ValueError("similarity_metric must be 'jaccard' or 'dice'")

    # Preprocess names from the file into 2-grams
    updated_predicted_list = []
    all_records = []
    with open(unique_names_file, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if name:
                name_grams = get_2grams(name)
                all_records.append((name, name_grams))

    # Match each predicted 2-gram entry to the most similar real name
    best_matches = []
    for entry in predicted_2grams_list:
        best_name = None
        best_score = -1
        for name, name_grams in all_records:
            score = similarity_func(entry['filtered_two_grams'], name_grams)
            if score > best_score:
                best_score = score
                best_name = name
                best_name_grams = name_grams
        best_matches.append((best_name, best_score, entry['uid']))
        updated_filtered_2grams = [gram for gram in entry['filtered_two_grams'] if gram not in best_name_grams]
        updated_predicted_list.append({
            'uid': entry['uid'],
            'actual_two_grams': entry['actual_two_grams'],
            'filtered_two_grams': updated_filtered_2grams
        })

    return best_matches, updated_predicted_list


def find_most_likely_surnames(predicted_2grams_list, minCount, similarity_metric='jaccard', use_filtered_surnames=False):
    # Choose similarity function
    if similarity_metric == 'jaccard':
        similarity_func = jaccard_similarity
    elif similarity_metric == 'dice':
        similarity_func = dice_coefficient
    else:
        raise ValueError("similarity_metric must be 'jaccard' or 'dice'")

    # Load surname records
    all_records = []
    file_path = 'data/names/surname/Filtered_Names_2010Census.csv' if use_filtered_surnames else 'data/names/surname/Names_2010Census.csv'
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row['name']
                count = int(row['count'])
                if count >= minCount:
                    grams = get_2grams(name)
                    all_records.append((name.lower(), {g.lower() for g in grams}, count))

    best_matches = []
    updated_predicted_list = []

    for entry in predicted_2grams_list:
        best_name = None
        best_name_grams = None
        best_score = -1

        for (name, name_grams, _) in all_records:
            score = similarity_func(entry['filtered_two_grams'], name_grams)
            if score > best_score:
                best_score = score
                best_name = name
                best_name_grams = name_grams

        # Remove matched 2-grams from filtered_two_grams
        updated_filtered_2grams = [gram for gram in entry['filtered_two_grams'] if gram not in best_name_grams]

        best_matches.append((best_name, best_score, entry['uid']))
        updated_predicted_list.append({
            'uid': entry['uid'],
            'actual_two_grams': entry['actual_two_grams'],
            'filtered_two_grams': updated_filtered_2grams
        })

    return best_matches, updated_predicted_list

def greedy_reconstruction(result):
    reconstructed_results = []

    for entry in result:
        uid = entry["uid"]
        two_grams = entry["filtered_two_grams"]

        # Step 1: Build a simple directed graph (DiGraph)
        G = nx.DiGraph()
        for two_gram in two_grams:
            G.add_edge(two_gram[0], two_gram[1])

        if nx.is_directed_acyclic_graph(G):
            # Use NetworkX's built-in longest path for DAGs
            path = nx.dag_longest_path(G)
            # Reconstruct string from path
            reconstructed = path[0] + ''.join(path[1:])

        else:
            # Fallback to DFS (Depth-First Search) if graph is not a DAG
            longest_reconstruction = ""

            def dfs(node, visited_edges, current_string):
                nonlocal longest_reconstruction
                if len(current_string) > len(longest_reconstruction):
                    longest_reconstruction = current_string

                for neighbor in G.successors(node):
                    edge = (node, neighbor)
                    if edge not in visited_edges:
                        visited_edges.add(edge)
                        dfs(neighbor, visited_edges, current_string + neighbor)
                        visited_edges.remove(edge)

            for two_gram in two_grams:
                dfs(two_gram[1], {(two_gram[0], two_gram[1])}, two_gram[0] + two_gram[1])

            reconstructed = longest_reconstruction

        reconstructed_results.append({
            "uid": uid,
            "identifier": reconstructed
        })

    return reconstructed_results

def fuzzy_reconstruction_approach(result):
    print("\n🔄 Reconstructing results using fuzzy matching...")
    best_matches_given_name, updated_result = find_most_likely_given_name(result, similarity_metric='dice')
    best_matches_surnames, updated_result = find_most_likely_surnames(predicted_2grams_list=updated_result, minCount=10, similarity_metric='dice', use_filtered_surnames=True)
    best_matches_birthday = find_most_likely_birthday(updated_result, similarity_metric='dice')

    reconstructed = [
        (given[0], surname[0], birthday[0], given[2])
        for given, surname, birthday in zip(best_matches_given_name, best_matches_surnames, best_matches_birthday)
    ]
    return reconstructed

def create_identifier_column_dynamic(df, components):
    col_series = [df[col].astype(str).str.replace('/', '', regex=False) for col in components]
    return pd.Series([''.join(vals) for vals in zip(*col_series)]).str.lower()

def reidentification_analysis(df_1, df_2, merge_on, len_not_reidentified, method_name=None, save_path=None):
    merged = pd.merge(
        df_1,
        df_2,
        on=merge_on,
        how='inner',
        suffixes=('_df1', '_df2')
    )

    total_reidentified = len(merged)
    total_not_reidentified = len_not_reidentified

    print(f"\n🔍 Reidentification Analysis:")
    print(f"Total Reidentified Individuals: {total_reidentified}")
    print(f"Total Not Reidentified Individuals: {total_not_reidentified}")

    if total_not_reidentified > 0:
        reidentification_rate = (total_reidentified / total_not_reidentified) * 100
        print(f"Reidentification Rate: {reidentification_rate:.2f}%")
    else:
        reidentification_rate = None
        print("No not reidentified individuals to analyze.")

    # Save if requested
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

        # Save merged reidentified individuals
        merged.to_csv(f"{save_path}/result_{method_name}.csv", index=False)

        # Save a summary txt file alongside CSV
        summary_path = os.path.splitext(save_path)[0] + "/summary_" + method_name + ".txt"
        with open(summary_path, "w") as f:
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
    elapsed_gma, elapsed_data_preparation, elapsed_hyperparameter_optimization,
    elapsed_model_training, elapsed_application_to_encoded_data,
    elapsed_refinement_and_reconstruction, elapsed_total, output_dir="dea_runtime_logs"
):
    os.makedirs(output_dir, exist_ok=True)
    # Output file path
    output_file = os.path.join(output_dir, "dea_runtime_log.txt")

    # Write the values to the file
    with open(output_file, "w") as f:
        f.write("DEA Runtime (in minutes):\n")
        f.write(f"Graph Matching Attack: {to_minutes(elapsed_gma)}m\n")
        f.write(f"Data Preparation: {to_minutes(elapsed_data_preparation)}m\n")
        f.write(f"Hyperparameter Optimization: {to_minutes(elapsed_hyperparameter_optimization)}m\n")
        f.write(f"Model Training: {to_minutes(elapsed_model_training)}m\n")
        f.write(f"Application to Encoded Data: {to_minutes(elapsed_application_to_encoded_data)}m\n")
        f.write(f"Refinement and Reconstruction: {to_minutes(elapsed_refinement_and_reconstruction)}m\n")
        f.write(f"Total Runtime: {to_minutes(elapsed_total)}m\n")
