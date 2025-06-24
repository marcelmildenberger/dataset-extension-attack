import numpy as np
import torch
def label_to_tensor(label, allTwoGrams):
        label_vector = np.zeros(len(allTwoGrams), dtype=np.float32)
        for gram in label:
            if gram in allTwoGrams:
                index = allTwoGrams.index(gram)
                label_vector[index] = 1
        return torch.tensor(label_vector)

def bit_string_to_tensor(bit_string):
    bit_string_array = np.array([int(bit) for bit in bit_string], dtype=np.float32)
    return torch.tensor(bit_string_array)