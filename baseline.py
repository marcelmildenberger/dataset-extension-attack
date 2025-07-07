import pandas as pd
import os
from collections import Counter
from utils import precision_recall_f1, extract_two_grams
import argparse

def analyze_2gram_baseline(file_path):
    # Load the data
    df = pd.read_csv(file_path, sep='\t')

    # Drop UID columns (case-insensitive)
    non_uid_cols = [col for col in df.columns if 'uid' not in col.lower()]
    df['full_entry'] = df[non_uid_cols].astype(str).agg(''.join, axis=1)

    # Calculate average and max entry length
    entry_lengths = df['full_entry'].apply(len)
    avg_length = int(round(entry_lengths.mean()))
    max_length = int(entry_lengths.max())

    # Get all 2-grams and their frequencies
    all_2grams = df['full_entry'].apply(extract_two_grams).sum()
    two_gram_counts = Counter(all_2grams)

    # Number of 2-grams that fit in the average length
    n = avg_length - 1

    # Get top-n most frequent 2-grams
    top_n_2grams = [gram for gram, _ in two_gram_counts.most_common(n)]

    # Generate true and predicted sets of 2-grams
    true_2grams = df['full_entry'].apply(extract_two_grams)

    total_precision = total_recall = total_f1 = 0.0

    for entry in true_2grams:
        precision, recall, f1 = precision_recall_f1(entry, top_n_2grams)
        total_precision += precision
        total_recall += recall
        total_f1 += f1

    # Save the results
    output_file = os.path.splitext(file_path)[0] + '_analysis.txt'
    with open(output_file, 'w') as f:
        f.write(f"Used columns: {non_uid_cols}\n")
        f.write(f"Average entry length: {avg_length}\n")
        f.write(f"Maximum entry length: {max_length}\n")
        f.write(f"Top {n} 2-grams: {top_n_2grams}\n")
        f.write(f"Precision: {total_precision / len(df):.4f}\n")
        f.write(f"Recall: {total_recall / len(df):.4f}\n")
        f.write(f"F1 Score: {total_f1 / len(df):.4f}\n")

    print(f"\n✅ Analysis complete. Results saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-gram baseline analysis on a dataset.")
    parser.add_argument('--file', type=str, help='Path to the TSV file for analysis')

    args = parser.parse_args()

    if args.file and os.path.isfile(args.file):
        analyze_2gram_baseline(args.file)
    else:
        file_path = input("Enter path to your TSV dataset: ").strip()
        if not os.path.isfile(file_path):
            print("❌ File not found.")
        else:
            analyze_2gram_baseline(file_path)
