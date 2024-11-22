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
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning import Trainer
from dataset_process import DataProcessor, get_project_paths
from training import BandgapTrainer
from matgl.utils.training import ModelLightningModule

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
        
        # Modified training parameters for fine-tuning
        self.training_params = {
            'num_epochs': kwargs.get('num_epochs', 5),  # Reduced epochs for fine-tuning
            'batch_size': kwargs.get('batch_size', 128),
            'learning_rate': kwargs.get('learning_rate', 1e-5),  # Reduced learning rate
            'accelerator': kwargs.get('accelerator', 'cpu'),
            'early_stopping_patience': kwargs.get('early_stopping_patience', 5),
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
        """Setup training loggers."""
        loggers = []
        # TensorBoard logger
        tensorboard_logger = TensorBoardLogger(
            save_dir=str(self.working_dir),
            name="finetune_logs",
            version="tensorboard"
        )
        loggers.append(tensorboard_logger)

        # CSV logger
        csv_logger = CSVLogger(
            save_dir=str(self.working_dir),
            name="finetune_logs",
            version="csv"
        )
        loggers.append(csv_logger)
        
        return loggers
        
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
            
            # # 3. Initialize base model
            # trainer = BandgapTrainer(
            #     working_dir=str(self.working_dir),
            #     config=self.training_params
            # )
            # trainer.setup_model(processor.element_list)
            
            
            # # 4. Load pre-trained weights
            # trainer.model.load_state_dict(prev_weights)
            
            # # 5. Create Lightning module for fine-tuning
            # lit_module = ModelLightningModule(
            #     model=trainer.model,
            #     lr=self.training_params['learning_rate']
            # )
                        
            # 3. Create lightning module
            lit_module = ModelLightningModule(
                model=self.model,
                lr=self.config['learning_rate']
            )
            # 6. Setup Lightning trainer with callbacks
            pl_trainer = Trainer(
                max_epochs=self.training_params['num_epochs'],
                accelerator=self.training_params['accelerator'],
                callbacks=self._setup_callbacks(),
                logger=self._setup_logger(),
                enable_progress_bar=True
            )
            
            # 7. Fine-tune the model
            self.logger.info("Starting fine-tuning...")
            pl_trainer.fit(
                model=lit_module,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader
            )
            
            # 8. Save fine-tuned model
            save_dir = self.working_dir / "checkpoints"
            save_dir.mkdir(exist_ok=True)
            model_path = save_dir / "finetuned_model.pt"
            torch.save(lit_module.model.state_dict(), model_path)
            self.logger.info(f"Fine-tuned model saved to {model_path}")
            
            # 9. Evaluate on test set
            test_results = pl_trainer.test(model=lit_module, dataloaders=test_loader)
            results_file = self.working_dir / 'finetuning_test_results.json'
            with open(results_file, 'w') as f:
                json.dump(test_results[0], f, indent=4)
            
            # 10. Plot learning curves
            self.plot_finetuning_curves()
            
            duration = time.time() - start_time
            self.logger.info(f"Fine-tuning completed in {duration:.2f} seconds")
            
            # Store the trainer for later use if needed
            self.trainer = trainer
            return trainer
            
        except Exception as e:
            self.logger.error(f"Error during fine-tuning: {str(e)}")
            raise

    def plot_finetuning_curves(self):
        """Plot and save fine-tuning learning curves."""
        csv_log_dir = self.working_dir / "finetune_logs" / "csv"
        version_dirs = [d for d in csv_log_dir.iterdir() if d.is_dir()]
        if not version_dirs:
            self.logger.warning("No training logs found to plot.")
            return
            
        latest_version = max(version_dirs, key=lambda d: int(d.name.split("_")[-1]))
        metrics_path = latest_version / "metrics.csv"
        
        try:
            metrics = pd.read_csv(metrics_path)
            
            plt.figure(figsize=(12, 6))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot training and validation loss
            epochs = range(1, len(metrics["train_loss"].dropna()) + 1)
            ax1.plot(epochs, metrics["train_loss"].dropna(), 
                    label='Train Loss', marker='o')
            ax1.plot(epochs, metrics["val_loss"].dropna(), 
                    label='Validation Loss', marker='o')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Fine-tuning Loss Curves')
            ax1.legend()
            
            # Plot training and validation MAE
            ax2.plot(epochs, metrics["train_MAE"].dropna(), 
                    label='Train MAE', marker='o')
            ax2.plot(epochs, metrics["val_MAE"].dropna(), 
                    label='Validation MAE', marker='o')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.set_title('Fine-tuning MAE Curves')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(
                self.working_dir / "finetuning_curves.png",
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
            
            self.logger.info("Fine-tuning curves saved.")
            
        except Exception as e:
            self.logger.error(f"Error plotting fine-tuning curves: {str(e)}")

def main():
    paths = get_project_paths()
    
    # Initialize fine-tuner with previous training directory
    finetuner = FineTuner(
        working_dir=os.path.join(paths['output_dir'], 'finetuning'),
        prev_training_dir=paths['output_dir'],
        num_epochs=5,
        learning_rate=1e-5,
        batch_size=64
    )
    
    trainer = finetuner.run_finetuning(paths)

if __name__ == "__main__":
    main()