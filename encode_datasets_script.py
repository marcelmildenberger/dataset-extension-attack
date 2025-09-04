#!/usr/bin/env python3
"""
Dataset Encoding Script

This script processes multiple TSV datasets and applies encoding using Bloom Filter (BF), 
Tabulation MinHash (TMH), or Two-Step Hash (TSH) encoders based on configuration.

Usage:
    python encode_datasets_script.py
"""

import json
import os
from pathlib import Path
from graphMatching.encoders.bf_encoder import BFEncoder
from graphMatching.encoders.tmh_encoder import TMHEncoder
from graphMatching.encoders.tsh_encoder import TSHEncoder
from utils import read_tsv, save_tsv

# Define the datasets to process
DATASETS = [
    "./data/datasets/fakename_1k.tsv",
    "./data/datasets/fakename_2k.tsv",
    "./data/datasets/fakename_5k.tsv",
    "./data/datasets/fakename_10k.tsv",
    "./data/datasets/fakename_20k.tsv",
    "./data/datasets/fakename_50k.tsv",
    "./data/datasets/euro_person_5k.tsv",
    "./data/datasets/euro_person.tsv",
    "./data/datasets/titanic_full.tsv"
]

# Define which encoders to use
ENCODERS = ['BF', 'TMH', 'TSH']

def load_config():
    """Load configuration from dea_config.json"""
    with open('dea_config.json', 'r') as f:
        config = json.load(f)
    
    enc_config = config['ENC_CONFIG']
    global_config = config['GLOBAL_CONFIG']
    global_config["Workers"] = os.cpu_count()
    
    return enc_config, global_config

def create_encoder(encoder_type, enc_config, global_config):
    """Create encoder instance based on type"""
    if encoder_type == 'BF':
        return BFEncoder(
            secret=enc_config["AliceSecret"], 
            filter_size=enc_config["AliceBFLength"],
            bits_per_feature=enc_config["AliceBits"], 
            ngram_size=enc_config["AliceN"], 
            diffusion=enc_config["AliceDiffuse"],
            eld_length=enc_config["AliceEldLength"], 
            t=enc_config["AliceT"]
        )
    elif encoder_type == 'TMH':
        return TMHEncoder(
            enc_config["AliceNHash"], 
            enc_config["AliceNHashBits"],
            enc_config["AliceNSubKeys"], 
            enc_config["AliceN"],
            enc_config["Alice1BitHash"],
            random_seed=enc_config["AliceSecret"], 
            verbose=global_config["Verbose"],
            workers=global_config["Workers"]
        )
    elif encoder_type == 'TSH':
        return TSHEncoder(
            enc_config["AliceNHashFunc"], 
            enc_config["AliceNHashCol"], 
            enc_config["AliceN"],
            enc_config["AliceRandMode"], 
            secret=enc_config["AliceSecret"],
            verbose=global_config["Verbose"], 
            workers=global_config["Workers"]
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

def process_dataset(dataset_path, encoder_type, enc_config, global_config):
    """Process a single dataset with the specified encoder"""
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_path} with {encoder_type} encoder")
    print(f"{'='*60}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset not found: {dataset_path}")
        return False
    
    try:
        # Read the dataset
        data, uids, header = read_tsv(dataset_path, skip_header=False)
        print(f"Loaded {len(data)} records from {dataset_path}")
        
        # Create encoder
        encoder = create_encoder(encoder_type, enc_config, global_config)
        print(f"Created {encoder_type} encoder")
        
        # Update header for encoding column - insert before the last column (uid)
        header_copy = header.copy()
        if encoder_type == 'BF':
            header_copy.insert(-1, "bloomfilter")
        if encoder_type == 'TMH':
            header_copy.insert(-1, "tabminhash")
        if encoder_type == 'TSH':
            header_copy.insert(-1, "twostephash")
        
        # Encode the data
        print(f"Encoding data with {encoder_type}...")
        _, encoded_data = encoder.encode_and_compare_and_append(
            data, uids, 
            metric=enc_config["AliceMetric"], 
            sim=True, 
            store_encs=global_config["SaveAliceEncs"]
        )
        
        # Create output filename
        dataset_path_obj = Path(dataset_path)
        output_filename = f"{dataset_path_obj.stem}_{encoder_type.lower()}_encoded.tsv"
        output_path = dataset_path_obj.parent / output_filename
        
        # Save encoded dataset
        save_tsv(encoded_data, str(output_path), header=header_copy, write_header=True)
        print(f"Saved encoded dataset: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {dataset_path} with {encoder_type}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to process all datasets with all encoders"""
    print("Dataset Encoding Script")
    print("=" * 60)
    
    # Load configuration
    try:
        enc_config, global_config = load_config()
        print("Configuration loaded successfully")
        print(f"Available datasets: {len(DATASETS)}")
        print(f"Available encoders: {', '.join(ENCODERS)}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    # Process each dataset with each encoder
    total_operations = len(DATASETS) * len(ENCODERS)
    completed_operations = 0
    successful_operations = 0
    
    print(f"\nStarting processing of {total_operations} operations...")
    
    for dataset_path in DATASETS:
        for encoder_type in ENCODERS:
            completed_operations += 1
            print(f"\nProgress: {completed_operations}/{total_operations}")
            
            success = process_dataset(dataset_path, encoder_type, enc_config, global_config)
            if success:
                successful_operations += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total operations: {total_operations}")
    print(f"Successful: {successful_operations}")
    print(f"Failed: {total_operations - successful_operations}")
    print(f"Success rate: {(successful_operations/total_operations)*100:.1f}%")
    
    if successful_operations == total_operations:
        print("\n🎉 All datasets processed successfully!")
    else:
        print(f"\n⚠️  {total_operations - successful_operations} operations failed. Check the logs above for details.")

if __name__ == "__main__":
    main()
