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
    def __init__(
        self,
        working_dir: str,
        prev_training_dir: str,  # Fix type annotation
        debug: bool = False,
        **kwargs
    ):
        self.working_dir = Path(working_dir)
        self.prev_training_dir = Path(prev_training_dir)  # Fix Path initialization
        self.debug = debug
        self.trainer = None
        
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
            # 1. Load previous training weights
            prev_model_path = self.prev_training_dir / "checkpoints" / "model.pt"
            
            self.logger.info(f"Loading previous training weights: {prev_model_path}")
            prev_weights = torch.load(prev_model_path)
            
            # 2. Setup data
            data_config = {
                'structures_dir': paths['structures_dir'],
                'file_path': paths['file_path'],
                'cutoff': 4.0,
                'batch_size': self.training_params['batch_size']
            }
            
            processor = DataProcessor(data_config)
            processor.load_data()
            dataset = processor.create_dataset(normalize=True)
            train_loader, val_loader, test_loader = processor.create_dataloaders()
            
            # 3. Initialize trainer
            trainer_config = {
                'batch_size': self.training_params['batch_size'],
                'num_epochs': self.training_params['num_epochs'],
                'learning_rate': self.training_params['learning_rate'],
                'accelerator': self.training_params['accelerator']
            }
            
            trainer = BandgapTrainer(
            working_dir=str(self.working_dir),
            config=trainer_config
            )
        
            # 4. Initialize model with same architecture as training
            trainer.setup_model(processor.element_list)
            trainer.model.load_state_dict(prev_weights)
            self.trainer = trainer
            
            # 5. Continue fine-tuning
            trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                element_types=processor.element_list
            )
            
            # Save results
            results = trainer.evaluate(test_loader)
            results_file = self.working_dir / 'test_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            
            trainer.plot_training_curves()
            
            duration = time.time() - start_time
            self.logger.info(f"Fine-tuning completed in {duration:.2f} seconds")
            
            return trainer
            
        except Exception as e:
            self.logger.error(f"Error during fine-tuning: {str(e)}")
            raise

def main():
    paths = get_project_paths()
    
    # Initialize with previous training directory
    finetuner = FineTuner(
        working_dir=os.path.join(paths['output_dir'], 'M3GNet_finetuning'),
        prev_training_dir=paths['output_dir'] # Point to previous training results
    )
    
    trainer = finetuner.run_finetuning(paths)

if __name__ == "__main__":
    main()