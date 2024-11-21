"""
Data processing module for crystal structure datasets.
Handles loading POSCAR files, creating graph representations,
and preparing data for machine learning tasks.
"""

from __future__ import annotations
import os
import warnings
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from pymatgen.core import Structure 
from pymatgen.io.vasp import Poscar
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader
from dgl.data.utils import split_dataset
from sklearn.preprocessing import StandardScaler

def get_project_paths():
    """
    Get and create necessary project directories.
    
    Returns:
        dict: Dictionary containing paths for:
            - structures_dir: Directory for processed structure files
            - file_path: Path to main data list CSV
            - output_dir: Directory for log files
            - diffusion_dir: Directory for diffusion-related files 
            - train_list: Path to training data list
            - filtered_train: Path to filtered training data
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    paths = {
        'structures_dir': os.path.join(root_dir, './data/structures'),
        'file_path': os.path.join(root_dir, './data/data_list.csv'),
        'output_dir': os.path.join(root_dir, 'logs'),
        'diffusion_dir': os.path.join(root_dir, 'diffusion'),
        'train_list': os.path.join(root_dir, 'train_bandgap_list.csv'),
        'filtered_train': os.path.join(root_dir, 'filtered_train_bandgap.csv'),
    }
    
    # Create directories if they don't exist
    for dir_path in paths.values():
        dir_name = os.path.dirname(dir_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
    
    return paths

class DataProcessor:
    """
    Processes crystal structure data and creates datasets for ML tasks.
    
    Handles loading of POSCAR files, conversion to graphs, data normalization,
    and creation of train/validation/test splits.
    """
    
    def __init__(self, config: dict):
        """
        Initialize data processor with configuration.
        
        Args:
            config (dict): Configuration containing:
                - structures_dir: Path to structure files
                - file_path: Path to data list
                - cutoff: Cutoff radius for graphs (default: 4.0)
                - batch_size: Batch size for loading (default: 32)
                - split_ratio: Train/val/test split (default: [0.7,0.1,0.2])
                - random_state: Random seed (default: 42)
        """
        self.structures_dir = config['structures_dir']
        self.file_path = config['file_path']
        self.cutoff = config.get('cutoff', 4.0)
        self.batch_size = config.get('batch_size', 32)
        self.split_ratio = config.get('split_ratio', [0.7, 0.1, 0.2])
        self.random_state = config.get('random_state', 42)
        
        # Data containers
        self.structures: List[Structure] = []
        self.bandgap_values: List[float] = []
        self.dataset: Optional[MGLDataset] = None
        self.element_list: List[str] = []

        # Split indices storage
        self.train_indices: Optional[np.ndarray] = None
        self.val_indices: Optional[np.ndarray] = None
        self.test_indices: Optional[np.ndarray] = None
        
    def read_poscar(self, file_path: str) -> Structure:
        """
        Read POSCAR file and convert to pymatgen Structure.
        
        Args:
            file_path (str): Path to POSCAR file
            
        Returns:
            Structure: Pymatgen Structure object
        """
        poscar = Poscar.from_file(file_path)
        return poscar.structure
    
    def load_data(self, bandgap_column: str = 'Bandgap_by_DFT') -> Tuple[List[Structure], List[float]]:
        """Load and process structure files and bandgap values."""
        print("Loading data from files...")
        df = pd.read_csv(self.file_path)
        sampled_df = df.sample(frac=1.0, random_state=self.random_state)
        
        for index, row in sampled_df.iterrows():
            try:
                file_name = row['FileName']
                struct = self.read_poscar(os.path.join(self.structures_dir, file_name))
                band_v = row[bandgap_column]
                
                self.structures.append(struct)
                self.bandgap_values.append(band_v)
            except Exception as e:
                print(f"Error processing file {file_name}: {str(e)}")
                
        print(f"Loaded {len(self.structures)} structures successfully")
        return self.structures, self.bandgap_values
    
    def normalize_bandgap_values(self) -> np.ndarray:
        """
        Normalize bandgap values using StandardScaler.
        
        Returns:
            np.ndarray: Normalized bandgap values
        """
        scaler = StandardScaler()
        normalized_values = scaler.fit_transform(np.array(self.bandgap_values).reshape(-1, 1))
        return normalized_values.flatten()
    
    def create_dataset(self, normalize: bool = True) -> MGLDataset:
        """Create graph dataset from structures and bandgap values."""
        if not self.structures or not self.bandgap_values:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # Extract unique elements
        self.element_list = get_element_list(self.structures)
        
        # Initialize graph converter
        converter = Structure2Graph(
            element_types=self.element_list,
            cutoff=self.cutoff
        )
        
        # Apply normalization if requested
        values = self.normalize_bandgap_values() if normalize else self.bandgap_values
        
        # Create and return dataset
        self.dataset = MGLDataset(
            structures=self.structures,
            converter=converter,
            labels={"bandgap": values}
        )
        
        return self.dataset
    
    def create_dataloaders(self) -> Tuple[MGLDataLoader, MGLDataLoader, MGLDataLoader]:
        """Create data loaders and store split indices."""
        if self.dataset is None:
            raise ValueError("Dataset not created. Call create_dataset() first.")
            
        # Split dataset into train/val/test
        train_data, val_data, test_data = split_dataset(
            self.dataset,
            frac_list=self.split_ratio,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Store split indices
        self.train_indices = train_data.indices
        self.val_indices = val_data.indices
        self.test_indices = test_data.indices
        
        # Create data loaders
        train_loader, val_loader, test_loader = MGLDataLoader(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            batch_size=self.batch_size,
            num_workers=1
        )
        
        print(f"Created dataloaders with sizes: Train={len(train_data)}, "
              f"Val={len(val_data)}, Test={len(test_data)}")
              
        return train_loader, val_loader, test_loader
    
    def save_split_indices(self, output_dir: str) -> None:
        """
        Save train/validation/test split indices to CSV files.
        
        Args:
            output_dir (str): Directory to save split files
        """
        # Verify splits exist
        if any(x is None for x in [self.train_indices, self.val_indices, self.test_indices]):
            raise ValueError("Split indices not available. Call create_dataloaders() first.")
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Load original data
        df = pd.read_csv(self.file_path)
        
        # Create subset dataframes
        train_df = df.iloc[self.train_indices]
        val_df = df.iloc[self.val_indices]
        test_df = df.iloc[self.test_indices]
        
        # Save to CSV files
        train_df.to_csv(os.path.join(output_dir, 'train_data_list.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, 'valid_data_list.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test_data_list.csv'), index=False)
        
        print(f"\nSplit indices saved to {output_dir}")
        print(f"Train set: {len(train_df)} samples")
        print(f"Validation set: {len(val_df)} samples")
        print(f"Test set: {len(test_df)} samples")

def main():
    """Main execution function to demonstrate the data processing pipeline."""
    
    paths = get_project_paths()
    
    # Create configuration
    config = {
        'structures_dir': paths['structures_dir'],
        'file_path': paths['file_path'],
        'output_dir': paths['output_dir'],
        'cutoff': 4.0,
        'batch_size': 16,
        'split_ratio': [0.6, 0.1, 0.3],
        'random_state': 42
    }
    
    # Initialize processor
    processor = DataProcessor(config)
    
    # Load and process data
    processor.load_data()
    
    # Create normalized dataset
    dataset = processor.create_dataset(normalize=True)
    
    # Create dataset and dataloaders
    dataset = processor.create_dataset(normalize=True)
    train_loader, val_loader, test_loader = processor.create_dataloaders()
    
    # Save split indices
    processor.save_split_indices('output/splits')
    
    print("Data processing completed successfully!")

if __name__ == "__main__":
    main()