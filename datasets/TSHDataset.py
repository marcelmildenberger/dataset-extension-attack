import numpy as np
import torch
from torch.utils.data import Dataset

class TSHDataset(Dataset):
    def __init__(self, data, isLabeled=False, all_two_grams=None, max_length=None):
        self.isLabeled = isLabeled
        self.allTwoGrams = all_two_grams
        self.data = data
        self.max_length = max_length
        self.hash_tensors = self.data['twostephash'].apply(lambda row: self.hash_list_to_tensor(list(row)))

        if self.isLabeled:
            # For reidentified data, extract labels (2-grams) from values except last two columns
            self.data['label'] = self.data.apply(lambda row: self.extract_two_grams("".join(row.iloc[:-2].astype(str))), axis=1)
            self.label_tensors = self.data['label'].apply(lambda row: self.label_to_tensor(row))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.isLabeled:
            return self.hash_tensors[idx], self.label_tensors[idx]
        else:
            return self.hash_tensors[idx]

    def hash_list_to_tensor(self, hash_list):
        hash_array = [int(entry) for entry in hash_list]
        hash_tensor = np.zeros(self.max_length, dtype=np.float32)
        for val in hash_array:
            hash_tensor[val-1] = hash_tensor[val-1] + 1
        return torch.tensor(hash_tensor)

    def extract_two_grams(self, input_string):
        input_string_preprocessed = input_string.replace('"', '').replace('.', '').replace('/', '').strip()
        input_string_lower = input_string_preprocessed.lower()
        return [input_string_lower[i:i+2] for i in range(len(input_string_lower) - 1) if ' ' not in input_string_lower[i:i+2]]

    def label_to_tensor(self, label_two_grams):
        label_vector = np.zeros(len(self.allTwoGrams), dtype=np.float32)
        for gram in label_two_grams:
            if gram in self.allTwoGrams:
                index = self.allTwoGrams.index(gram)
                label_vector[index] = 1
        return torch.tensor(label_vector)
