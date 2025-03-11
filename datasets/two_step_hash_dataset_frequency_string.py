import numpy as np
import torch
from datasets.dataset_utils import *
from torch.utils.data import Dataset

class TwoStepHashDatasetFrequencyString(Dataset):
    def __init__(self, data, is_labeled=False, all_two_grams=None, frequency_string_length=None, dev_mode=False):
        self.isLabeled = is_labeled
        self.allTwoGrams = all_two_grams
        self.frequencyStringLength = frequency_string_length
        self.devMode = dev_mode

        self.hashTensors = data['twostephash'].apply(lambda row: self.hash_list_to_tensor(list(row)))

        if self.isLabeled:
            self.labelTensors = data.apply(lambda row: label_to_tensor(extract_two_grams("".join(row.iloc[:-2].astype(str))), self.allTwoGrams))

        if self.devMode:
            self.data = data
            if self.isLabeled:
                self.data['label'] = self.data.apply(lambda row: extract_two_grams("".join(row.iloc[:-2].astype(str))), axis=1)

    def __len__(self):
        return len(self.labelTensors)

    def __getitem__(self, idx):
        if self.isLabeled:
            return self.hashTensors[idx], self.labelTensors[idx]
        else:
            return self.hashTensors[idx]

    def hash_list_to_tensor(self, hash_list):
        hash_array = [int(entry) for entry in hash_list]
        hash_tensor = np.zeros(self.max_length, dtype=np.float32)
        for val in hash_array:
            hash_tensor[val-1] = hash_tensor[val-1] + 1
        return torch.tensor(hash_tensor)
