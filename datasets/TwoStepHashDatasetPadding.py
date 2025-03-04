import numpy as np
import torch
from torch.utils.data import Dataset

class TwoStepHashDatasetPadding(Dataset):
    def __init__(self, data, isLabeled=False, all_two_grams=None, max_length=None):
        self.isLabeled = isLabeled
        self.allTwoGrams = all_two_grams
        self.data = data
        self.max_length = max_length

        if self.isLabeled:
            # For reidentified data, extract labels (2-grams) from values except last two columns
            self.data['label'] = self.data.apply(lambda row: self.extract_two_grams("".join(row.iloc[:-2].astype(str))), axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tsh_encoding = self.data.iloc[idx]['twostephash']
        # Parse the string (e.g., "{273928, 1785355, ...}") into a list of integers
        hash_tensor = self.hash_list_to_tensor(list(tsh_encoding))

        if self.isLabeled:
            label = self.data.iloc[idx]['label']
            label_tensor = self.label_to_tensor(label)
            return hash_tensor, label_tensor
        else:
            return hash_tensor

    def hash_list_to_tensor(self, hash_list):
        hash_array = np.array(hash_list, dtype=np.float32)
        if self.max_length is not None:
            if len(hash_array) < self.max_length:
                pad_width = self.max_length - len(hash_array)
                hash_array = np.pad(hash_array, (0, pad_width), mode='constant', constant_values=0)
            else:
                hash_array = hash_array[:self.max_length]
        return torch.tensor(hash_array)

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
