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
        # Base directories
        self.working_dir = Path(working_dir)
        self.prev_training_dir = Path(prev_training_dir)
        
        # Create organized subdirectories
        self.checkpoints_dir = self.working_dir / "checkpoints"
        self.metrics_dir = self.working_dir / "logs/finetune/metrics"  
        self.logs_dir = self.working_dir / "logs/finetune"  
        self.results_dir = self.working_dir / "results"
        self.plots_dir = self.working_dir / "plots"
        
        # Create all directories
        for dir_path in [self.checkpoints_dir, self.metrics_dir, self.logs_dir,
                        self.results_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.debug = debug
        self.model = None
        self.trainer = None
        
        # Modified training parameters for fine-tuning
        self.config = {
            'batch_size': kwargs.get('batch_size', 128),
            'num_epochs': kwargs.get('num_epochs', 5),  # Reduced epochs for fine-tuning
            'learning_rate': kwargs.get('learning_rate', 1e-5),  # Reduced learning rate
            'accelerator': kwargs.get('accelerator', 'cpu')
        }
        
        self._setup_logging()
        self.save_config()
    
    def save_config(self):
        """Save configuration to JSON file."""
        config_path = self.results_dir / 'finetune_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        self.logger.info(f"Configuration saved to {config_path}")
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.logs_dir / 'finetune.log'
        
        logging.basicConfig(
            level=logging.DEBUG if self.debug else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    # def run_finetuning(self, paths: Dict):
    #     """Run fine-tuning process using pretrained model."""
    #     start_time = time.time()
    #     self.logger.info("Starting fine-tuning process...")
        
    #     try:
    #         # 1. Load previous training weights
    #         prev_model_path = self.prev_training_dir / "checkpoints" / "base_model.pt"
    #         self.logger.info(f"Loading previous training weights: {prev_model_path}")
    #         prev_weights = torch.load(prev_model_path)
            
    #         # 2. Setup data
    #         data_config = {
    #             'structures_dir': paths['structures_dir'],
    #             'file_path': paths['file_path'],
    #             'cutoff': 4.0,
    #             'batch_size': self.config['batch_size']
    #         }
            
    #         processor = DataProcessor(data_config)
    #         processor.load_data()
    #         dataset = processor.create_dataset(normalize=True)
    #         train_loader, val_loader, test_loader = processor.create_dataloaders()
            
    #         # 3. Initialize trainer
    #         trainer_config = {
    #             'batch_size': self.config['batch_size'],
    #             'num_epochs': self.config['num_epochs'],
    #             'learning_rate': self.config['learning_rate'],
    #             'accelerator': self.config['accelerator']
    #         }

    #         # Setup trainer with correct directories
    #         trainer = BandgapTrainer(
    #             working_dir=str(self.working_dir),
    #             config=trainer_config,
    #             debug=self.debug
    #         )
        
    #         # Override trainer's logging directories to use finetune paths
    #         trainer.metrics_dir = self.metrics_dir
    #         trainer.logs_dir = self.logs_dir
            
    #         # 4. Setup logger with organized directory
    #         logger = CSVLogger(
    #             save_dir=str(self.logs_dir),
    #             name="metrics"
    #         )
            
    #         # 5. Initialize model with same architecture as training
    #         trainer.setup_model(processor.element_list)
    #         trainer.model.load_state_dict(prev_weights)
    #         self.trainer = trainer
            
    #         # 6. Continue fine-tuning
    #         trainer.train(
    #             train_loader=train_loader,
    #             val_loader=val_loader,
    #             element_types=processor.element_list
    #         )
            
    #         # 7. Save fine-tuned model with distinctive name
    #         model_path = self.checkpoints_dir / "finetuned_model.pt"
    #         torch.save(trainer.model.state_dict(), model_path)
            
    #         # 8. Evaluate model
    #         results = trainer.evaluate(test_loader)
    #         results_file = self.results_dir / 'finetune_results.json'
    #         with open(results_file, 'w') as f:
    #             json.dump(results, f, indent=4)
            
    #         # 9. Plot learning curves
    #         trainer.plot_learning_curves()
            
    #         duration = time.time() - start_time
    #         self.logger.info(f"Fine-tuning completed in {duration:.2f} seconds")
    #         return results
            
    #     except Exception as e:
    #         self.logger.error(f"Error during fine-tuning: {str(e)}")
    #         raise

    def run_finetuning(self, paths: Dict):
            """Run fine-tuning process using pretrained model."""
            start_time = time.time()
            self.logger.info("Starting fine-tuning process...")
    
            try:
                # 1. Load previous training weights
                prev_model_path = self.prev_training_dir / "checkpoints" / "base_model.pt"
                self.logger.info(f"Loading previous training weights: {prev_model_path}")
                prev_weights = torch.load(prev_model_path)
            
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
                    working_dir=str(self.working_dir),
                    config=trainer_config,
                    debug=self.debug
                )
                
                # 4. Override trainer's directories completely
                trainer.metrics_dir = self.metrics_dir
                trainer.logs_dir = self.logs_dir
                trainer.plots_dir = self.plots_dir  # 确保覆盖plots目录
                trainer.results_dir = self.results_dir  # 确保覆盖results目录
                trainer.checkpoints_dir = self.checkpoints_dir  # 确保覆盖checkpoints目录
                
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
                
                # 7. Save fine-tuned model with distinctive name - 现在使用trainer的checkpoints_dir
                model_path = trainer.checkpoints_dir / "finetuned_model.pt"
                torch.save(trainer.model.state_dict(), model_path)
                
                # 8. Evaluate model - 结果会保存到trainer的results_dir
                results = trainer.evaluate(test_loader)
                
                # 9. Plot learning curves - 直接使用trainer的方法，会使用正确的路径
                trainer.plot_learning_curves()
                
                duration = time.time() - start_time
                self.logger.info(f"Fine-tuning completed in {duration:.2f} seconds")
                return results
                
            except Exception as e:
                self.logger.error(f"Error during fine-tuning: {str(e)}")
                raise

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