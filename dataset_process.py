from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import csv
from pymatgen.io.vasp import Poscar
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader
from dgl.data.utils import split_dataset

warnings.simplefilter("ignore")

def read_poscar(file_path):
    """Read POSCAR file and return structure"""
    try:
        poscar = Poscar.from_file(file_path)
        return poscar.structure
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def process_dataset(max_samples=100, structures_dir='proccessed_data'):
    """Process dataset with small sample size for testing"""
    # Set paths
    file_path = 'normalized_4704_shuffled_data_w_noise_v5.csv'
    
    # Read CSV file with limited samples
    try:
        df = pd.read_csv(file_path)
        # Reset index before sampling to avoid index issues
        df = df.reset_index(drop=True)
        df = df.sample(n=min(max_samples, len(df)), random_state=1)
        df = df.reset_index(drop=True)  # Reset index again after sampling
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        raise
    
    # Load structures and bandgaps
    structures = []
    bandgap_values = []
    processed_indices = []
    
    for idx, row in df.iterrows():
        file_path = os.path.join(structures_dir, row['FileName'])
        if os.path.exists(file_path):
            struct = read_poscar(file_path)
            if struct is not None:
                structures.append(struct)
                bandgap_values.append(row['Bandgap_by_DFT'])
                processed_indices.append(idx)
    
    if not structures:
        raise ValueError("No valid structures found!")
    
    print(f"Successfully loaded {len(structures)} structures")
    
    # Create dataset
    elem_list = get_element_list(structures)
    converter = Structure2Graph(element_types=elem_list, cutoff=4.0)
    
    mp_dataset = MGLDataset(
        structures=structures,
        converter=converter,
        labels={"bandgap": bandgap_values},
    )
    
    # Split dataset
    train_data, val_data, test_data = split_dataset(
        mp_dataset,
        frac_list=[0.6, 0.2, 0.2],
        shuffle=True,
        random_state=42,
    )
    
    # Create loaders
    train_loader, val_loader, test_loader = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        batch_size=8,
        num_workers=1,
    )
    
    # Save only processed data
    processed_df = df.iloc[processed_indices]
    processed_df.to_csv('filtered_train_bandgap_small.csv', index=False)
    
    return train_loader, val_loader, test_loader, elem_list

if __name__ == "__main__":
    train_loader, val_loader, test_loader, elem_list = process_dataset()
    
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")