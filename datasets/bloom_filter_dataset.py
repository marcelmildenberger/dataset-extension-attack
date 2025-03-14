from datasets.dataset_utils import *
from torch.utils.data import Dataset

class BloomFilterDataset(Dataset):
    def __init__(self, data, is_labeled=False, all_two_grams=None, dev_mode=False):
        self.isLabeled = is_labeled
        self.allTwoGrams = all_two_grams
        self.devMode = dev_mode

        self.bitStringTensors = data['bloomfilter'].apply(lambda row: bit_string_to_tensor(list(row)))

        if self.isLabeled:
            self.labelTensors = data.apply(lambda row: label_to_tensor(extract_two_grams("".join(row.iloc[:-2].astype(str))), self.allTwoGrams), axis=1)

        if dev_mode:
            self.data = data
            if self.isLabeled:
                self.data['label'] = self.data.apply(lambda row: extract_two_grams("".join(row.iloc[:-2].astype(str))), axis=1)

    def __len__(self):
        return len(self.labelTensors)

    def __getitem__(self, idx):
         if self.isLabeled:
            return self.bitStringTensors[idx], self.labelTensors[idx]
         else:
            return self.bitStringTensors[idx]