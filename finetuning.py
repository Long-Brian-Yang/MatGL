from __future__ import annotations
import os
import json
import time
import logging
import warnings
from pathlib import Path
from typing import Dict
import torch
import pandas as pd
import matplotlib.pyplot as plt
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
        pretrained_checkpoint: str,
        freeze_base_layers: bool = True,
        debug: bool = False,
        **kwargs
    ):
        """
        Initialize the FineTuner for bandgap prediction.

        Args:
            working_dir (str): Directory where all outputs will be saved.
            debug: If True, sets logging level to DEBUG. Defaults to False.
            **kwargs: Additional keyword arguments for training configuration.
        """
        self.working_dir = Path(working_dir)
        self.debug = debug
        self.pretrained_checkpoint = pretrained_checkpoint
        self.freeze_base_layers = freeze_base_layers

        # Define directories for checkpoints, logs, and results
        self.checkpoints_dir = self.working_dir / "checkpoints"
        self.logs_dir = self.working_dir / "logs"
        self.results_dir = self.working_dir / "results"

        # Create directories if they do not exist
        for dir_path in [
                self.checkpoints_dir,
                self.logs_dir,
                self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.trainer = None
        self.lit_module = None

        # Training configuration with default values
        self.config = {
            'batch_size': kwargs.get('batch_size', 32),
            'num_epochs': kwargs.get('num_epochs', 10),
            'learning_rate': kwargs.get('learning_rate', 1e-4),
            'fine_tune_lr': kwargs.get('fine_tune_lr', 1e-5),
            'accelerator': kwargs.get('accelerator', 'cpu'),
            'split_ratio': kwargs.get('split_ratio', [0.7, 0.1, 0.2]),
            'random_state': kwargs.get('random_state', 42),
            'weight_decay': kwargs.get('weight_decay', 1e-5)
        }

        # Setup logging and save configuration
        self.setup_logging()
        self.save_config()

    def setup_logging(self) -> None:
        """
        Setup logging configuration for the training process.
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

    def save_config(self) -> None:
        """
        Save the training configuration to a JSON file for future reference.
        """
        config_path = self.results_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

    def setup_model(self, element_list) -> None:
        """
        Setup the M3GNet model and PyTorch Lightning trainer.

        Args:
            element_list (list): List of element types used in the model.
        """
        try:
            self.logger.info("Setting up M3GNet model...")
            # 1. Load the pretrained model
            pretrained_model = matgl.load_model(self.pretrained_checkpoint)

            # 2. Create a new M3GNet model with the same element types
            self.model = M3GNet(
                element_types=element_list,
                is_intensive=True,
                readout_type="set2set"
            )

            # 3. Transfer the pretrained weights to the new model
            self.model.load_state_dict(
                pretrained_model.state_dict(), strict=False)
            self.logger.info("Transferred pretrained weights to new model")

            # 4. Freeze base layers if required
            if self.freeze_base_layers:
                for name, param in self.model.named_parameters():
                    if "readout" not in name:
                        param.requires_grad = False
                    self.logger.info("Froze base layers")

            # # 5. Define optimizer with different learning rates
            # self.lit_module = ModelLightningModule(
            #     model=self.model,
            #     learning_rate=self.config['learning_rate']
            # )
            # 5. Define optimizer with different learning rates
            optimizer = torch.optim.Adam([
                {"params": [p for n, p in self.model.named_parameters() if "readout" not in n],
                 "lr": self.config['fine_tune_lr']},
                {"params": [p for n, p in self.model.named_parameters() if "readout" in n],
                 "lr": self.config['learning_rate']}
            ])

            self.lit_module = ModelLightningModule(model=self.model)

            def configure_optimizers(self):
                return self.optimizer

            def on_train_epoch_end(self):
                pass
            self.lit_module.configure_optimizers = configure_optimizers.__get__(
                self.lit_module)
            self.lit_module.on_train_epoch_end = on_train_epoch_end.__get__(
                self.lit_module)
            self.lit_module.optimizer = optimizer
            logger = CSVLogger(
                save_dir=str(self.logs_dir),
                name="",
                version=""
            )

            # Initialize the PyTorch Lightning Trainer
            self.trainer = pl.Trainer(
                max_epochs=self.config['num_epochs'],
                accelerator=self.config['accelerator'],
                logger=logger,
                inference_mode=False,
                log_every_n_steps=1
            )

            self.logger.info("Model setup completed")

        except Exception as e:
            self.logger.error(f"Error in model setup: {str(e)}")
            raise

    def run_training(self, paths: Dict) -> list:
        """
        Execute the training process for bandgap prediction.

        Args:
            paths (Dict): Dictionary containing paths to data files.

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

    def plot_training_curves(self, metrics: pd.DataFrame) -> None:
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

    # Initialize the FineTuner with custom configuration
    trainer = FineTuner(
        working_dir=os.path.join(paths['output_dir']),
        pretrained_checkpoint="M3GNet-MP-2021.2.8-PES",
        freeze_base_layers=True,
        num_epochs=1000,
        learning_rate=1e-4,
        fine_tune_lr=1e-5,
        batch_size=32
    )

    results = trainer.run_training(paths)
    print(f"Test results: {results}")


if __name__ == "__main__":
    main()