from datasets.dataset_utils import *
from torch.utils.data import Dataset

class BloomFilterDatasetReversed(Dataset):
    def __init__(self, data, is_labeled=False, all_two_grams=None, dev_mode=False, reversed=False):
        self.isLabeled = is_labeled
        self.allTwoGrams = all_two_grams
        self.devMode = dev_mode
        self.reveresed = reversed

        self.twoGramTensors = data.apply(lambda row: label_to_tensor(extract_two_grams("".join(row.iloc[:-2].astype(str))), self.allTwoGrams), axis=1)
        self.uids = data['uid']

        if self.isLabeled:
            self.labelTensors = data['bloomfilter'].apply(lambda row: bit_string_to_tensor(list(row)))

    def __len__(self):
        return len(self.twoGramTensors)

    def __getitem__(self, idx):
         if self.isLabeled:
            return self.twoGramTensors[idx], self.labelTensors[idx], self.uids[idx]
         else:
            return self.twoGramTensors[idx], self.uids[idx]