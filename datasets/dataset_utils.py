import numpy as np
import torch

def extract_two_grams(input_string, remove_spaces=False):
    input_string_preprocessed = input_string.replace('"', '').replace('.', '').replace('/', '').strip()
    if(remove_spaces):
        input_string_preprocessed = input_string_preprocessed.replace(' ', '')
    input_string_lower = input_string_preprocessed.lower()  # Normalize to lowercase for consistency
    return [input_string_lower[i:i+2] for i in range(len(input_string_lower)-1) if ' ' not in input_string_lower[i:i+2]]

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