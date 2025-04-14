import numpy as np
import torch
from datasets.dataset_utils import *
from torch.utils.data import Dataset

class TwoStepHashDataset(Dataset):
    def __init__(self, data, is_labeled=False, all_integers=None, dev_mode=False, all_two_grams=None):
        self.isLabeled = is_labeled
        self.allIntegers = all_integers
        self.allTwoGrams = all_two_grams
        self.devMode = dev_mode

        self.hashTensors = data['twostephash'].apply(lambda row: self.hash_list_to_tensor(list(row)))
        self.uids = data['uid']

        if self.isLabeled:
            self.labelTensors = data.apply(lambda row: label_to_tensor(extract_two_grams("".join(row.iloc[:-2].astype(str))), self.allTwoGrams),  axis=1)

        if dev_mode:
            self.data = data
            if self.isLabeled:
                self.data['label'] = self.data.apply(lambda row: extract_two_grams("".join(row.iloc[:-2].astype(str)), self.allTwoGrams), axis=1)

    def __len__(self):
        return len(self.hashTensors)

    def __getitem__(self, idx):
        if self.isLabeled:
            return self.hashTensors[idx], self.labelTensors[idx], self.uids[idx]
        else:
            return self.hashTensors[idx], self.uids[idx]

    def hash_list_to_tensor(self, hash_list):
        hash_array = np.zeros(len(self.allIntegers), dtype=np.float32)
        for val in hash_list:
            if val in self.allIntegers:
                index = self.allIntegers.index(val)
                hash_array[index] = 1
        return torch.tensor(hash_array)


