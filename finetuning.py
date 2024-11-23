from __future__ import annotations
import os
import json
import time
import logging
import warnings
import shutil
from pathlib import Path
from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt
import torch
import lightning as pl
from pytorch_lightning.loggers import CSVLogger
import matgl
from matgl.models import M3GNet
from matgl.utils.training import ModelLightningModule

from dataset_process import DataProcessor, get_project_paths

class FineTuner:
    def __init__(
        self,
        working_dir: str,
        debug: bool = False,
        **kwargs
    ):
        """Initialize fine-tuner for bandgap prediction."""
        self.working_dir = Path(working_dir)
        
        # Create directories
        self.checkpoints_dir = self.working_dir / "checkpoints"
        self.logs_dir = self.working_dir / "logs"
        self.results_dir = self.working_dir / "results"
        
        for dir_path in [self.checkpoints_dir, self.logs_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.debug = debug
        self.model = None
        self.trainer = None
        self.lit_module = None
        
        # Training configuration
        self.config = {
            'batch_size': kwargs.get('batch_size', 32),
            'num_epochs': kwargs.get('num_epochs', 2),
            'learning_rate': kwargs.get('learning_rate', 1e-4),
            'accelerator': kwargs.get('accelerator', 'cpu'),
            'split_ratio': kwargs.get('split_ratio', [0.6, 0.1, 0.3]),
            'random_state': kwargs.get('random_state', 42)
        }
        
        self._setup_logging()
        self._save_config()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.logs_dir / 'train.log'
        
        logging.basicConfig(
            level=logging.DEBUG if self.debug else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _save_config(self):
        """Save configuration to JSON file."""
        config_path = self.results_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

    def setup_model(self, element_list):
        """Setup M3GNet model for bandgap prediction."""
        try:
            self.logger.info("Setting up M3GNet model...")
            
            # Create M3GNet model
            self.model = M3GNet(
                element_types=element_list,
                is_intensive=True,
                readout_type="set2set"
            )
            
            # Create lightning module
            self.lit_module = ModelLightningModule(
                model=self.model
            )
            
            # Setup trainer
            logger = CSVLogger(
                save_dir=str(self.logs_dir),
                name="",
                version=""
            )
            
            self.trainer = pl.Trainer(
                max_epochs=self.config['num_epochs'],
                accelerator=self.config['accelerator'],
                logger=logger,
                inference_mode=False,
                log_every_n_steps=1 
            )
            
            self.logger.info("Model setup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in model setup: {str(e)}")
            raise

    def run_training(self, paths: Dict):
        """Run bandgap prediction training."""
        start_time = time.time()
        self.logger.info("Starting training process...")

        try:
            # 1. Prepare data
            data_config = {
                'structures_dir': paths['structures_dir'],
                'file_path': paths['file_path'],
                'batch_size': self.config['batch_size'],
                'split_ratio': self.config['split_ratio'],
                'random_state': self.config['random_state']
            }
            
            # Process data
            processor = DataProcessor(data_config)
            processor.load_data()
            dataset = processor.create_dataset(normalize=True)
            train_loader, val_loader, test_loader = processor.create_dataloaders()
            
            # 2. Setup model
            self.setup_model(processor.element_list)
            
            # 3. Train
            self.logger.info("Starting training...")
            self.trainer.fit(
                model=self.lit_module,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader
            )
            
            # 4. Save model
            model_save_path = self.checkpoints_dir
            self.lit_module.model.save(str(model_save_path))
            self.logger.info(f"Model saved to {model_save_path}")
            
            # 5. Test
            test_results = self.trainer.test(
                model=self.lit_module,
                dataloaders=test_loader
            )
            
            # Save results
            results_file = self.results_dir / 'metrics.csv'
            pd.DataFrame(test_results).to_csv(results_file, index=False)
            
            # 6. Plot and save training curves
            metrics = pd.read_csv(self.logs_dir / "metrics.csv")
            self._plot_training_curves(metrics)
            
            duration = time.time() - start_time
            self.logger.info(f"Training completed in {duration:.2f} seconds")
            return test_results
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
        
        finally:
            # Cleanup
            for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin",
                      "state_attr.pt", "labels.json"):
                try:
                    os.remove(fn)
                except FileNotFoundError:
                    pass

    def _plot_training_curves(self, metrics):
        """Plot and save training curves."""
        try:
            plt.figure(figsize=(10, 6))
            metrics["train_MAE"].dropna().plot(label='Training MAE')
            metrics["val_MAE"].dropna().plot(label='Validation MAE')
            
            plt.xlabel('Iterations')
            plt.ylabel('MAE')
            plt.legend()
            
            plot_path = self.logs_dir / "training_curve.png"
            plt.savefig(
                plot_path,
                facecolor='w',
                bbox_inches="tight",
                pad_inches=0.3,
                transparent=True
            )
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting training curves: {str(e)}")

def main():
    paths = get_project_paths()
    
    trainer = FineTuner(
        working_dir=os.path.join(paths['output_dir']),
        num_epochs=2,
        learning_rate=1e-4,
        batch_size=32,
        split_ratio=[0.6, 0.1, 0.3],
        random_state=42
    )
    
    results = trainer.run_training(paths)
    print(f"Test results: {results}")

if __name__ == "__main__":
    main()

