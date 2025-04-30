import csv
from typing import Sequence
from hashlib import md5

import torch
from tqdm import tqdm

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


