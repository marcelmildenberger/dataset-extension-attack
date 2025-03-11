import numpy as np
import torch
from datasets.dataset_utils import *
from torch.utils.data import Dataset

class TwoStepHashDatasetPadding(Dataset):
    def __init__(self, data, isLabeled=False, all_two_grams=None, max_set_size=None, dev_mode=False):
        self.isLabeled = isLabeled
        self.allTwoGrams = all_two_grams
        self.devMode = dev_mode
        self.maxSetSize = max_set_size

        self.hashTensors = data['twostephash'].apply(lambda row: self.hash_list_to_tensor(list(row)))

        if self.isLabeled:
            self.labelTensors = data.apply(lambda row: label_to_tensor(extract_two_grams("".join(row.iloc[:-2].astype(str))), self.allTwoGrams))

        if dev_mode:
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
        hash_array = np.array(hash_list, dtype=np.float32)
        if self.max_length is not None:
            if len(hash_array) < self.max_length:
                pad_width = self.max_length - len(hash_array)
                hash_array = np.pad(hash_array, (0, pad_width), mode='constant', constant_values=0)
            else:
                hash_array = hash_array[:self.max_length]
        return torch.tensor(hash_array)


