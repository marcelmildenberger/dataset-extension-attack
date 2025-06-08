import csv
import json
from typing import Sequence
from hashlib import md5
from collections import defaultdict
from groq import Groq
import os
import pandas as pd
import torch
from tqdm import tqdm
from dotenv import load_dotenv

from torch.utils.data import Subset

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

def reconstruct_words(ngrams):
    words = []
    used = set()

    for ngram in ngrams:
        if ngram in used:
            continue

        word = ngram
        used.add(ngram)
        extended = True

        while extended:
            extended = False
            for next_ngram in ngrams:
                if next_ngram in used:
                    continue

                if word[-1] == next_ngram[0]:
                    word += next_ngram[1]
                    used.add(next_ngram)
                    extended = True
                    break

        words.append(word)

    return words


def extract_two_grams(input_string, remove_spaces=False):
    input_string_preprocessed = input_string.replace('"', '').replace('.', '').replace('/', '').strip()
    if(remove_spaces):
        input_string_preprocessed = input_string_preprocessed.replace(' ', '')
    input_string_lower = input_string_preprocessed.lower()  # Normalize to lowercase for consistency
    return [input_string_lower[i:i+2] for i in range(len(input_string_lower)-1) if ' ' not in input_string_lower[i:i+2]]

def dice_coefficient(set1: set, set2: set) -> float:
    if isinstance(set1, list):
        set1 = set(set1)
    if isinstance(set2, list):
        set2 = set(set2)
    if not set1 and not set2:
        return 1.0  # both empty sets → full similarity
    intersection = len(set1 & set2)
    return round((2 * intersection) / (len(set1) + len(set2)),4)

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


def greedy_reconstruct_ngrams(ngrams: set[str]) -> list[str]:
    ngrams = set(ngrams)
    adjacency = defaultdict(list)

    # Build directed graph: a→b if a[1] == b[0]
    for a in ngrams:
        for b in ngrams:
            if a != b and a[1] == b[0]:
                adjacency[a].append(b)

    def dfs(path, used):
        current = path[-1]
        if current not in adjacency:
            return [''.join([p[0] for p in path]) + path[-1][1]]
        results = []
        for neighbor in adjacency[current]:
            if neighbor not in used:
                results += dfs(path + [neighbor], used | {neighbor})
        if not results:
            return [''.join([p[0] for p in path]) + path[-1][1]]
        return results

    all_results = set()
    for ng in ngrams:
        all_results.update(dfs([ng], {ng}))

    return sorted(all_results, key=len, reverse=True)

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

        Return **only** the reconstructed values in **valid JSON format**, as a list of dictionaries. Each dictionary must include the UID and the inferred values, as shown below:

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


def reidentification_analysis(df_1, df_2, merge_on, len_not_reidentified):
    """
    Analyze the reidentification results and print statistics.
    """
    merged = pd.merge(
    df_1,
    df_2,
    on=merge_on,
    how='inner',
    suffixes=('_df1', '_df2')
)

    total_reidentified = len(merged)
    total_not_reidentified = len(df_not_reidentified)

    print(f"\n🔍 Reidentification Analysis:")
    print(f"Total Reidentified Individuals: {total_reidentified}")
    print(f"Total Not Reidentified Individuals: {total_not_reidentified}")

    if total_not_reidentified > 0:
        reidentification_rate = (total_reidentified / total_not_reidentified) * 100
        print(f"Reidentification Rate: {reidentification_rate:.2f}%")
    else:
        print("No not reidentified individuals to analyze.")

    return merged


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



