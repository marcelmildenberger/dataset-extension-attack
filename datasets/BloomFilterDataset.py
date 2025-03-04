import numpy as np
import torch
from torch.utils.data import Dataset

class BloomFilterDataset(Dataset):
    def __init__(self, data, isLabeled=False, all_two_grams=None):
        self.isLabeled = isLabeled
        self.allTwoGrams = all_two_grams
        self.data = data
        if self.isLabeled:
            # For reidentified data, extract labels (2-grams) from values except last two columns which are encoding and uid
            self.data['label'] = self.data.apply(lambda row: self.extract_two_grams("".join(row.iloc[:-2].astype(str))), axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bloom_filter = self.data.iloc[idx]['bloomfilter']
        bloom_filter_tensor = self.bloomfilter_to_tensor(bloom_filter)

        if self.isLabeled:
            label = self.data.iloc[idx]['label']
            label_tensor = self.label_to_tensor(label)
            return bloom_filter_tensor, label_tensor
        else:
            # For unlabeled data (not_reidentified_individuals.tsv), just return the Bloom Filter
            return bloom_filter_tensor

    def extract_two_grams(self, input_string):
        input_string_preprocessed = input_string.replace('"', '').replace('.', '').replace('/', '').strip()
        input_string_lower = input_string_preprocessed.lower()  # Normalize to lowercase for consistency
        return [input_string_lower[i:i+2] for i in range(len(input_string_lower)-1) if ' ' not in input_string_lower[i:i+2]]

    def bloomfilter_to_tensor(self, bloom_filter_str):
        bloom_filter_array = np.array([int(bit) for bit in bloom_filter_str], dtype=np.float32)
        return torch.tensor(bloom_filter_array)

    def label_to_tensor(self, label_two_grams):
        label_vector = np.zeros(len(self.allTwoGrams), dtype=np.float32)

        # Set 1 for the 2-grams present in the name
        for gram in label_two_grams:
            if gram in self.allTwoGrams:
                index = self.allTwoGrams.index(gram)
                label_vector[index] = 1

        return torch.tensor(label_vector)