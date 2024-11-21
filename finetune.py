"""
Enhanced M3GNet fine-tuning module with simplified parameter management.
"""

from __future__ import annotations
import os
import shutil
import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import matplotlib.pyplot as plt
from matgl import load_model
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from dataset_process import DataProcessor, get_project_paths
from training import BandgapTrainer

class FineTuner:
    """Enhanced fine-tuning manager with simplified configuration."""
    
    def __init__(
        self,
        working_dir: str,
        debug: bool = False,
        **kwargs
    ):
        """
        Initialize fine-tuning manager.
        
        Args:
            working_dir: Working directory path
            debug: Enable debug logging
            **kwargs: Optional training parameters
        """
        self.working_dir = Path(working_dir)
        self.debug = debug
        
        # Set training parameters with defaults
        self.training_params = {
            'num_epochs': kwargs.get('num_epochs', 2),
            'batch_size': kwargs.get('batch_size', 128),
            'learning_rate': kwargs.get('learning_rate', 1e-4),
            'accelerator': kwargs.get('accelerator', 'cpu'),
            'early_stopping_patience': kwargs.get('early_stopping_patience', 10),
            'checkpoint_monitor': kwargs.get('checkpoint_monitor', 'val_loss')
        }
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        self.working_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.working_dir / 'finetune.log'
        
        logging.basicConfig(
            level=logging.DEBUG if self.debug else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _setup_callbacks(self) -> list:
        """Setup training callbacks."""
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.working_dir / "checkpoints",
            filename="model-{epoch:02d}-{val_loss:.2f}",
            monitor=self.training_params['checkpoint_monitor'],
            mode='min',
            save_top_k=3,
            save_last=True
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping callback
        early_stop_callback = EarlyStopping(
            monitor=self.training_params['checkpoint_monitor'],
            patience=self.training_params['early_stopping_patience'],
            mode='min'
        )
        callbacks.append(early_stop_callback)
        
        return callbacks
        
    def _setup_logger(self):
        """Setup training logger."""
        return TensorBoardLogger(
            save_dir=str(self.working_dir),
            name="training_logs"
        )
        
    def run_finetuning(self, paths: Dict) -> BandgapTrainer:
        """
        Run complete fine-tuning process.
        
        Args:
            paths: Dictionary containing project paths
            
        Returns:
            BandgapTrainer: Trained model trainer
        """
        start_time = time.time()
        self.logger.info("Starting fine-tuning process...")
        
        try:
            # Load pretrained model
            model_name = "M3GNet-MP-2021.2.8-PES"
            self.logger.info(f"Loading pretrained model: {model_name}")
            pretrained_model = load_model(model_name).model
            
            # Setup data configuration
            data_config = {
                'structures_dir': paths['structures_dir'],
                'file_path': paths['file_path'],
                'cutoff': 4.0,
                'batch_size': self.training_params['batch_size']
            }
            
            # Process data
            processor = DataProcessor(data_config)
            processor.load_data()
            dataset = processor.create_dataset(normalize=True)
            train_loader, val_loader, test_loader = processor.create_dataloaders()
            
            # Setup trainer
            trainer_config = {
                'batch_size': self.training_params['batch_size'],
                'num_epochs': self.training_params['num_epochs'],
                'learning_rate': self.training_params['learning_rate'],
                'accelerator': self.training_params['accelerator']
            }
            
            # Initialize trainer with pretrained model
            trainer = BandgapTrainer(
                working_dir=str(self.working_dir),
                config=trainer_config
            )
            trainer.model = pretrained_model
            
            # Train model
            trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                element_types=processor.element_list
            )
            
            # Evaluate model
            results = trainer.evaluate(test_loader)
            
            # Save results
            results_file = self.working_dir / 'test_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
                
            # Plot training curves
            trainer.plot_training_curves()
            
            duration = time.time() - start_time
            self.logger.info(f"Fine-tuning completed in {duration:.2f} seconds")
            
            return trainer
            
        except Exception as e:
            self.logger.error(f"Error during fine-tuning: {str(e)}")
            raise

def main():
    """Main execution function."""
    # Get project paths
    paths = get_project_paths()
    
    # Basic usage
    finetuner = FineTuner(working_dir="output")
    trainer = finetuner.run_finetuning(paths)
    
    # Advanced usage with custom parameters
    """
    finetuner = FineTuner(
        working_dir="output",
        num_epochs=200,
        batch_size=256,
        learning_rate=1e-5,
        debug=True
    )
    trainer = finetuner.run_finetuning(paths)
    """

if __name__ == "__main__":
    main()