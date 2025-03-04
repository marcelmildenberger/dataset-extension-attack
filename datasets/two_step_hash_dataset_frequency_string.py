import numpy as np
import torch
from datasets.dataset_utils import *
from torch.utils.data import Dataset

class TwoStepHashDatasetFrequencyString(Dataset):
    def __init__(self, data, isLabeled=False, all_two_grams=None, max_length=None):
        self.isLabeled = isLabeled
        self.allTwoGrams = all_two_grams
        self.data = data
        self.max_length = max_length
        self.hash_tensors = self.data['twostephash'].apply(lambda row: self.hash_list_to_tensor(list(row)))

        if self.isLabeled:
            # For reidentified data, extract labels (2-grams) from values except last two columns
            self.data['label'] = self.data.apply(lambda row: extract_two_grams("".join(row.iloc[:-2].astype(str))), axis=1)
            self.label_tensors = self.data['label'].apply(lambda row: label_to_tensor(row, self.allTwoGrams))

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
