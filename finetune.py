from __future__ import annotations
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pytorch_lightning.loggers import CSVLogger
import lightning as pl
from matgl.utils.training import ModelLightningModule

from dataset_process import DataProcessor, get_project_paths
from training import BandgapTrainer
class FineTuner:
    def __init__(
        self,
        working_dir: str,
        prev_training_dir: str,
        debug: bool = False,
        **kwargs
    ):
        self.working_dir = Path(working_dir)
        self.prev_training_dir = Path(prev_training_dir)
        self.debug = debug
        self.model = None
        self.trainer = None
        
        self.config = {
            'batch_size': kwargs.get('batch_size', 128),
            'num_epochs': kwargs.get('num_epochs', 5),
            'learning_rate': kwargs.get('learning_rate', 1e-5),
            'accelerator': kwargs.get('accelerator', 'cpu')
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
        
    def run_finetuning(self, paths: Dict):
        start_time = time.time()
        self.logger.info("Starting fine-tuning process...")
        
        try:
            # 1. Load previous training weights
            prev_model_path = self.prev_training_dir / "checkpoints" / "model.pt"
            self.logger.info(f"Loading previous training weights: {prev_model_path}")
            prev_weights= torch.load(prev_model_path)
            
            # 2. Setup data
            data_config = {
                'structures_dir': paths['structures_dir'],
                'file_path': paths['file_path'],
                'cutoff': 4.0,
                'batch_size': self.config['batch_size']
            }
            
            processor = DataProcessor(data_config)
            processor.load_data()
            dataset = processor.create_dataset(normalize=True)
            train_loader, val_loader, test_loader = processor.create_dataloaders()
            
            # 3. Initialize trainer
            trainer_config = {
                'batch_size': self.config['batch_size'],
                'num_epochs': self.config['num_epochs'],
                'learning_rate': self.config['learning_rate'],
                'accelerator': self.config['accelerator']
            }

            trainer = BandgapTrainer(
            working_dir=str(self.working_dir / "finetuning"),
            config=trainer_config
            )
            
            # 4. Setup logger - using same pattern as training.py
            logger = CSVLogger(
                save_dir=str(self.working_dir),
                name="M3GNet_finetuning"
            )
            
            # 5. Initialize model with same architecture as training
            trainer.setup_model(processor.element_list)
            trainer.model.load_state_dict(prev_weights)
            self.trainer = trainer
            
            # 6. Continue fine-tuning
            trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                element_types=processor.element_list
            )
            
            # 7. Save fine-tuned model
            save_dir = self.working_dir / "finetuning" / "checkpoints"
            save_dir.mkdir(exist_ok=True, parents=True)
            model_path = save_dir / "finetuned_model.pt"
            torch.save(trainer.model.state_dict(), model_path)

            # 8. Evaluate and save results
            results = trainer.evaluate(test_loader)
            results_file = self.working_dir / "finetuning" / 'finetuning_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            
            print(f"Test MAE: {results['test_MAE']:.4f}")
            print(f"Test RMSE: {results['test_RMSE']:.4f}")
            
            # 9. Plot training curves
            self.plot_training_curves()
            
            duration = time.time() - start_time
            self.logger.info(f"Fine-tuning completed in {duration:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error during fine-tuning: {str(e)}")
            raise
            
    def plot_finetuning_curves(self):
        """Plot and save training/validation MAE curves."""
        finetune_dir = self.working_dir / "finetuning" / "M3GNet_finetuning"
        if not finetune_dir.exists():
            self.logger.warning(f"Finetuning directory not found: {finetune_dir}")
            return
            
        version_dirs = [d for d in finetune_dir.iterdir() 
                      if d.is_dir() and d.name.startswith("version_")]
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
        plt.title('Fine-tuning Progress')
        plt.legend()

        plot_path = self.working_dir / "finetuning" / "finetuning_curve.png"
        plt.savefig(
            plot_path,
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        print("\nFine-tuning curve saved.")

def main():
    paths = get_project_paths()
    
    finetuner = FineTuner(
        working_dir=os.path.join(paths['output_dir'], 'finetuning'),
        prev_training_dir=paths['output_dir'],
        num_epochs=5,
        learning_rate=1e-5
    )
    
    finetuner.run_finetuning(paths)

if __name__ == "__main__":
    main()