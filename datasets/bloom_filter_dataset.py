from datasets.dataset_utils import *
from torch.utils.data import Dataset

class BloomFilterDataset(Dataset):
    def __init__(self, data, isLabeled=False, all_two_grams=None):
        self.isLabeled = isLabeled
        self.allTwoGrams = all_two_grams
        self.data = data
        if self.isLabeled:
            # For reidentified data, extract labels (2-grams) from values except last two columns which are encoding and uid
            self.data['label'] = self.data.apply(lambda row: extract_two_grams("".join(row.iloc[:-2].astype(str))), axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bloom_filter = self.data.iloc[idx]['bloomfilter']
        bloom_filter_tensor = bit_string_to_tensor(bloom_filter)

        if self.isLabeled:
            label = self.data.iloc[idx]['label']
            label_tensor = label_to_tensor(label, self.allTwoGrams)
            return bloom_filter_tensor, label_tensor
        else:
            return bloom_filter_tensor