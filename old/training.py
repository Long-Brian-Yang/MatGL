# training.py
from __future__ import annotations
import logging
import time 
import json
import warnings
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import lightning as pl
import matplotlib.pyplot as plt
import torch
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from matgl.models import M3GNet
from matgl.utils.training import ModelLightningModule
from dataset_process import DataProcessor, get_project_paths

warnings.simplefilter("ignore")

class BandgapTrainer:
    def __init__(
        self,
        working_dir: str,
        config: Optional[Dict] = None,
        debug: bool = False,
    ):
        # Base directory
        self.working_dir = Path(working_dir)
        self.debug = debug
        
        # Create organized subdirectories
        self.checkpoints_dir = self.working_dir / "checkpoints"
        self.metrics_dir = self.working_dir / "logs/training/metrics" 
        self.logs_dir = self.working_dir / "logs/training"
        self.results_dir = self.working_dir / "results"
        self.plots_dir = self.working_dir / "plots"
        
        # Create all directories
        for dir_path in [self.checkpoints_dir, self.metrics_dir, self.logs_dir,
                        self.results_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
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
        
        self._setup_logging()
        self.save_config()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.logs_dir / 'training.log'
        
        logging.basicConfig(
            level=logging.DEBUG if self.debug else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def save_config(self):
        """Save configuration to JSON file."""
        config_path = self.results_dir / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        self.logger.info(f"Configuration saved to {config_path}")

    def setup_model(self, element_types: list):
        """Initialize M3GNet model with given element types."""
        # Save element types for future reference
        element_types_path = self.results_dir / 'element_types.json'
        with open(element_types_path, 'w') as f:
            json.dump(element_types, f)
            
        self.model = M3GNet(
            element_types=element_types,
            is_intensive=True,
            readout_type="set2set"
        )
        self.logger.info("Model initialized successfully")

    def train(self, train_loader, val_loader, element_types: list):
        """Train the model using provided data loaders."""
        start_time = time.time()
        self.logger.info("Starting training process...")
        
        try:
            if self.model is None:
                self.setup_model(element_types)

            lit_module = ModelLightningModule(
                model=self.model, 
                lr=self.config['learning_rate']
            )

            # Setup logger with organized directory
            logger = CSVLogger(
                save_dir=str(self.metrics_dir),
                name="metrics"
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
            
            # Save final model
            model_path = self.checkpoints_dir / "base_model.pt"
            torch.save(lit_module.model.state_dict(), model_path)
            self.logger.info(f"Final model saved to {model_path}")
            
            duration = time.time() - start_time
            self.logger.info(f"Training completed in {duration:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def evaluate(self, test_loader) -> Dict:
        """Evaluate model on test dataset."""
        self.logger.info("Starting model evaluation...")
        
        if not self.trainer:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            results = self.trainer.test(dataloaders=test_loader)[0]
            
            # Save evaluation results
            results_path = self.results_dir / "evaluation_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
                
            self.logger.info(f"Test MAE: {results['test_MAE']:.4f}")
            self.logger.info(f"Test RMSE: {results['test_RMSE']:.4f}")
            self.logger.info(f"Evaluation results saved to {results_path}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            raise
        
    def plot_learning_curves(self):
        """Plot and save training/validation MAE curves."""
        self.logger.info("Plotting learning curves...")
        
        try:
            metrics_file = list(self.metrics_dir.glob("metrics/version_*/metrics.csv"))
            if not metrics_file:
                self.logger.warning("No metrics file found")
                return
                
            metrics = pd.read_csv(metrics_file[-1])

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

            plot_path = self.plots_dir / "training_learning_curves.png"
            plt.savefig(
                plot_path,
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
            self.logger.info(f"Learning curves saved to {plot_path}")
            
        except Exception as e:
            self.logger.error(f"Error plotting learning curves: {str(e)}")
            raise

def main():
    paths = get_project_paths()
    
    # Setup data
    data_config = {
        'structures_dir': paths['structures_dir'],
        'file_path': paths['file_path'],
        'cutoff': 4.0,
        'batch_size': 128
    }
    
    processor = DataProcessor(data_config)
    processor.load_data()
    dataset = processor.create_dataset(normalize=True)
    train_loader, val_loader, test_loader = processor.create_dataloaders()
    
    # Initialize and run trainer
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
    
    # Train and evaluate
    trainer.train(train_loader, val_loader, processor.element_list)
    trainer.evaluate(test_loader)
    trainer.plot_learning_curves()

if __name__ == "__main__":
    main()
