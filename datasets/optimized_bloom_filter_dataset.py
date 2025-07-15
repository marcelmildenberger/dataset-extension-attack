import torch
import numpy as np
from torch.utils.data import Dataset
from datasets.dataset_utils import *
from utils import extract_two_grams

class OptimizedBloomFilterDataset(Dataset):
    """
    Optimized BloomFilter dataset with performance improvements:
    - Vectorized bit string conversion
    - Batch processing for labels
    - Reduced memory footprint
    - Efficient tensor operations
    """
    
    def __init__(self, data, is_labeled=False, all_two_grams=None, dev_mode=False, reversed=False):
        self.isLabeled = is_labeled
        self.allTwoGrams = all_two_grams
        self.devMode = dev_mode
        self.reversed = reversed

        # Vectorized bit string conversion for better performance
        self.bitStringTensors = self._vectorized_bit_conversion(data['bloomfilter'])
        self.uids = data['uid'].values

        if self.isLabeled:
            # Batch process all labels at once for efficiency
            self.labelTensors = self._batch_process_labels(data)

        if dev_mode:
            self.data = data
            if self.isLabeled:
                # Efficient label extraction for dev mode
                self.data['label'] = self._extract_labels_vectorized(data)

    def _vectorized_bit_conversion(self, bloom_filter_series):
        """
        Convert bloom filter bit strings to tensors using vectorized operations.
        This is much faster than using pandas apply().
        """
        # Convert to list first, then create tensor in one operation
        bit_arrays = [list(bf) for bf in bloom_filter_series]
        # Convert to float32 tensor in one operation
        return torch.tensor(bit_arrays, dtype=torch.float32)

    def _batch_process_labels(self, data):
        """
        Process all labels in batch for better performance.
        """
        # Combine text columns efficiently
        text_data = data.iloc[:, :-2].astype(str)
        combined_text = text_data.apply(''.join, axis=1)
        
        # Extract two-grams for all samples
        all_two_grams_batch = []
        for text in combined_text:
            grams = extract_two_grams(text)
            all_two_grams_batch.append(grams)
        
        # Convert to label tensors in batch
        label_tensors = []
        for grams in all_two_grams_batch:
            label_tensor = label_to_tensor(grams, self.allTwoGrams)
            label_tensors.append(label_tensor)
        
        return torch.stack(label_tensors)

    def _extract_labels_vectorized(self, data):
        """
        Vectorized label extraction for development mode.
        """
        text_data = data.iloc[:, :-2].astype(str)
        combined_text = text_data.apply(''.join, axis=1)
        return combined_text.apply(lambda x: extract_two_grams(x))

    def __len__(self):
        return len(self.bitStringTensors)

    def __getitem__(self, idx):
        if self.isLabeled:
            return self.bitStringTensors[idx], self.labelTensors[idx], self.uids[idx]
        else:
            return self.bitStringTensors[idx], self.uids[idx]

    @staticmethod
    def create_optimized_from_original(original_dataset):
        """
        Factory method to create an optimized dataset from the original.
        """
        return OptimizedBloomFilterDataset(
            data=original_dataset.data,
            is_labeled=original_dataset.isLabeled,
            all_two_grams=original_dataset.allTwoGrams,
            dev_mode=original_dataset.devMode,
            reversed=original_dataset.reversed
        )