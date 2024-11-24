from __future__ import annotations
import json
import warnings
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import lightning as pl
import matplotlib.pyplot as plt
import torch
from pytorch_lightning.loggers import CSVLogger
from matgl.models import M3GNet
from matgl.utils.training import ModelLightningModule
from dataset_process import DataProcessor, get_project_paths

warnings.simplefilter("ignore")

class BandgapTrainer:
    """
    Trainer class for bandgap prediction model.
    Handles model setup, training, evaluation and result visualization.
    """
    
    def __init__(
        self, 
        working_dir: str,
        config: Optional[Dict] = None
    ):
        """
        Initialize trainer with working directory and configuration.
        
        Args:
            working_dir: Directory for saving model outputs
            config: Optional configuration dictionary
        """
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = {
            'batch_size': 128,
            'num_epochs': 10,
            'learning_rate': 1e-4,
            'accelerator': 'cpu'
        }
        if config:
            self.config.update(config)

        self.model = None
        self.trainer = None
        self.save_config()

    def save_config(self):
        """Save configuration to JSON file in working directory."""

        config_path = self.working_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

    def setup_model(self, element_types: list):
        """
        Initialize M3GNet model with given element types.
        
        Args:
            element_types: List of chemical elements in dataset
        """
        
        self.model = M3GNet(
            element_types=element_types,
            is_intensive=True,
            readout_type="set2set"
        )

    def train(self, train_loader, val_loader, element_types: list):
        """
        Train the model using provided data loaders.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            element_types: List of chemical elements
        """

        if self.model is None:
            self.setup_model(element_types)

        lit_module = ModelLightningModule(
            model=self.model, 
            lr=self.config['learning_rate']
        )

        logger = CSVLogger(
            save_dir=str(self.working_dir), 
            name="M3GNet_training"
        )
        
        self.trainer = pl.Trainer(
            max_epochs=self.config['num_epochs'],
            accelerator=self.config['accelerator'],
            logger=logger
        )
        
        self.trainer.fit(
            model=lit_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        
        self.save_model(lit_module)

    def save_model(self, lit_module):
        """
        Save trained model weights to checkpoint directory.
        
        Args:
            lit_module: Trained lightning module
        """

        save_dir = self.working_dir / "checkpoints"
        save_dir.mkdir(exist_ok=True)
        model_path = save_dir / "model.pt"
        torch.save(lit_module.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def evaluate(self, test_loader) -> Dict:
        """
        Evaluate model on test dataset.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary containing test metrics
        """
                
        if not self.trainer:
            raise ValueError("Model not trained. Call train() first.")
        
        results = self.trainer.test(dataloaders=test_loader)[0]
        print(f"Test MAE: {results['test_MAE']:.4f}")
        print(f"Test RMSE: {results['test_RMSE']:.4f}")
        return results
        
    def plot_training_curves(self):
        """Plot and save training/validation MAE curves."""
        
        version_dirs = [d for d in (self.working_dir / "M3GNet_training").iterdir() if d.is_dir() and d.name.startswith("version_")]
        latest_version = max(version_dirs, key=lambda d: int(d.name.split("_")[-1]))
        
        metrics_path = latest_version / "metrics.csv"

        metrics = pd.read_csv(metrics_path)

        plt.figure(figsize=(10, 6))
        epochs = range(1, len(metrics["train_MAE"].dropna()) + 1)

        plt.plot(epochs, metrics["train_MAE"].dropna(), 
                 label='Train MAE', marker='o')
        plt.plot(epochs, metrics["val_MAE"].dropna(), 
                 label='Validation MAE', marker='o')

        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('Training Progress')
        plt.legend()

        plt.savefig(
            self.working_dir / "training_curve.png",
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        print("\nTraining curve saved.")

def process_data(config: Dict):
    """
    Process raw data into DataLoaders.
    
    Args:
        config: Data processing configuration
        
    Returns:
        Tuple of (processor, train_loader, val_loader, test_loader)
    """

    processor = DataProcessor(config)
    processor.load_data()
    dataset = processor.create_dataset(normalize=True)
    train_loader, val_loader, test_loader = processor.create_dataloaders()
    return processor, train_loader, val_loader, test_loader

def train_model(trainer: BandgapTrainer, train_loader, val_loader, processor):
    """
    Train model using processed data.
    
    Args:
        trainer: Initialized BandgapTrainer
        train_loader: Training data loader
        val_loader: Validation data loader
        processor: Data processor containing element information
    """

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        element_types=processor.element_list
    )

def evaluate_model(trainer: BandgapTrainer, test_loader):
    """
    Evaluate trained model on test set.
    
    Args:
        trainer: Trained BandgapTrainer
        test_loader: Test data loader
        
    Returns:
        Dictionary of test metrics
    """

    results = trainer.evaluate(test_loader)
    return results

def main():
    """Main execution function for training pipeline."""
    paths = get_project_paths()
    
    data_config = {
        'structures_dir': paths['structures_dir'],
        'file_path': paths['file_path'], 
        'cutoff': 4.0,
        'batch_size': 128
    }
    processor, train_loader, val_loader, test_loader = process_data(data_config)
    
    element_types = processor.element_list
    with open('element_types.json', 'w') as f:
        json.dump(element_types, f)

    trainer_config = {
        'batch_size': 128,
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'accelerator': 'cpu'
    }
    trainer = BandgapTrainer(
        working_dir=paths['output_dir'], 
        config=trainer_config
    )
    
    train_model(trainer, train_loader, val_loader, processor)
    results = evaluate_model(trainer, test_loader)
    
    results_file = trainer.working_dir / 'test_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Test results saved to {results_file}")

    # Plot training curves
    trainer.plot_training_curves()

if __name__ == "__main__":
    main()