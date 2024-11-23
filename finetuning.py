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

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

class FineTuner:
    def __init__(
        self,
        working_dir: str,
        debug: bool = False,
        **kwargs
    ):
        """
        Initialize the FineTuner for bandgap prediction.

        Args:
            working_dir (str): Directory where all outputs (logs, checkpoints, results) will be saved.
            debug (bool, optional): If True, sets logging level to DEBUG. Defaults to False.
            **kwargs: Additional configuration parameters such as batch_size, num_epochs, etc.
        """

        self.working_dir = Path(working_dir)

        # Define directories for checkpoints, logs, and results
        self.checkpoints_dir = self.working_dir / "checkpoints"
        self.logs_dir = self.working_dir / "logs"
        self.results_dir = self.working_dir / "results"

        # Create directories if they do not exist
        for dir_path in [self.checkpoints_dir, self.logs_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.debug = debug
        self.model = None
        self.trainer = None
        self.lit_module = None

        # Training configuration with default values
        self.config = {
            'batch_size': kwargs.get('batch_size', 32),
            'num_epochs': kwargs.get('num_epochs', 2),
            'learning_rate': kwargs.get('learning_rate', 1e-4),
            'accelerator': kwargs.get('accelerator', 'cpu'),
            'split_ratio': kwargs.get('split_ratio', [0.6, 0.1, 0.3]),
            'random_state': kwargs.get('random_state', 42)
        }

        # Setup logging and save configuration
        self._setup_logging()
        self._save_config()
    
    def _setup_logging(self):
        """
        Setup logging configuration to log messages to both a file and the console.
        """

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
        """
        Save the training configuration to a JSON file for future reference.
        """

        config_path = self.results_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

    def setup_model(self, element_list):
        """
        Setup the M3GNet model and PyTorch Lightning trainer.

        Args:
            element_list (list): List of element types used in the model.
        """

        try:
            self.logger.info("Setting up M3GNet model...")
            
            # Initialize the M3GNet model with specified elements
            self.model = M3GNet(
                element_types=element_list,
                is_intensive=True,
                readout_type="set2set"
            )
            
            # Wrap the model in a PyTorch Lightning module
            self.lit_module = ModelLightningModule(
                model=self.model
            )
            
            # Initialize CSVLogger to log training metrics
            logger = CSVLogger(
                save_dir=str(self.logs_dir),
                name="", # Name can be specified if needed
                version="" # Version can be specified if needed
            )

            # Initialize the PyTorch Lightning Trainer
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
        """
        Execute the training process for bandgap prediction.

        Args:
            paths (Dict): Dictionary containing project paths such as structures_dir and file_path.

        Returns:
            test_results (list): List of test metrics after training.
        """

        start_time = time.time()
        self.logger.info("Starting training process...")

        try:
            # 1. Prepare data configuration
            data_config = {
                'structures_dir': paths['structures_dir'],
                'file_path': paths['file_path'],
                'batch_size': self.config['batch_size'],
                'split_ratio': self.config['split_ratio'],
                'random_state': self.config['random_state']
            }

            # 2. Process data
            processor = DataProcessor(data_config)
            processor.load_data()
            dataset = processor.create_dataset(normalize=True)
            train_loader, val_loader, test_loader = processor.create_dataloaders()

            # 3. Setup model with element list from data processor
            self.setup_model(processor.element_list)

            # 4. Start training
            self.logger.info("Starting training...")
            self.trainer.fit(
                model=self.lit_module,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader
            )

            # 5. Save the trained model to the checkpoints directory
            model_save_path = self.checkpoints_dir
            self.lit_module.model.save(str(model_save_path))
            self.logger.info(f"Model saved to {model_save_path}")

            # 6. Evaluate the model on the test dataset
            test_results = self.trainer.test(
                model=self.lit_module,
                dataloaders=test_loader
            )

            # 7. Save test results to a CSV file in the results directory
            results_file = self.results_dir / 'metrics.csv'
            pd.DataFrame(test_results).to_csv(results_file, index=False)
            
            # 8. Plot and save training curves from the logged metrics
            metrics = pd.read_csv(self.logs_dir / "metrics.csv")
            self.plot_training_curves(metrics)
            
            # 9. Log the duration of the training process
            duration = time.time() - start_time
            self.logger.info(f"Training completed in {duration:.2f} seconds")
            return test_results

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

        finally:
            # 10. Cleanup temporary files if they exist
            for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin",
                      "state_attr.pt", "labels.json"):
                try:
                    os.remove(fn)
                except FileNotFoundError:
                    pass

    def plot_training_curves(self, metrics):
        """
        Plot and save the training and validation MAE curves.

        Args:
            metrics (pd.DataFrame): DataFrame containing training metrics.
        """
        try:
            plt.figure(figsize=(10, 6))
            
            # Plot Training MAE if it exists in metrics
            if "train_MAE" in metrics.columns:
                metrics["train_MAE"].dropna().plot(label='Training MAE')
            
            # Plot Validation MAE if it exists in metrics
            if "val_MAE" in metrics.columns:
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
    """
    Main function to initiate the fine-tuning process.
    """

    paths = get_project_paths()
    
    # Initialize the FineTuner with the working directory and configuration parameters
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

