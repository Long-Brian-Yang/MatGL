from __future__ import annotations
import os
import json
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import pandas as pd
import matplotlib.pyplot as plt
import lightning as pl
from pytorch_lightning.loggers import CSVLogger
from dgl.data.utils import split_dataset

import matgl
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_pes  # Use collate_fn_pes
from matgl.models import M3GNet
from matgl.utils.training import PotentialLightningModule
from pymatgen.core import Structure

warnings.simplefilter("ignore")


def get_project_paths() -> Dict[str, str]:
    """Get commonly used paths in the project.

    Returns:
        Dict[str, str]: Dictionary containing project paths.
    """
    base_dir = os.getcwd()
    paths = {
        'base_dir': base_dir,
        'structures_dir': os.path.join(base_dir, 'data', 'structures'),
        'file_path': os.path.join(base_dir, 'data', 'data_list.csv'),
        'output_dir': os.path.join(base_dir, 'output')
    }
    return paths


class DataProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.structures_dir = Path(config['structures_dir'])
        self.file_path = Path(config['file_path'])
        self.batch_size = config['batch_size']
        self.split_ratio = config['split_ratio']
        self.random_state = config['random_state']

        self.structures: List[Structure] = []
        self.labels: List[float] = []
        self.element_list: List[str] = []

    def load_data(self):
        """Load structures and labels data."""
        self.structures, self.labels = self._load_dataset()
        self.element_list = get_element_list(self.structures)
        logging.info(f"Loaded {len(self.structures)} structures.")

    def _load_dataset(self) -> Tuple[List[Structure], List[float]]:
        """Load dataset from file.

        Returns:
            Tuple[List[Structure], List[float]]: List of structures and list of labels.
        """
        # Read data from data_list.csv
        data_list = pd.read_csv(self.file_path)

        structures = []
        labels = []

        for idx, row in data_list.iterrows():
            structure_filename = row['FileName']  # Based on your CSV file column name
            label = row['Bandgap_by_DFT']  # The label column you want to use, e.g., 'Bandgap_by_DFT'

            # Check if the label is a valid number
            if pd.isnull(label):
                logging.warning(f"Sample {structure_filename} has missing label, skipping this sample.")
                continue

            structure_path = self.structures_dir / structure_filename
            if not structure_path.exists():
                logging.warning(f"File {structure_path} does not exist, skipping this structure.")
                continue

            # Read structure from VASP format file using pymatgen
            try:
                structure = Structure.from_file(structure_path)
                structures.append(structure)
                labels.append(float(label))  # Ensure the label is a float
            except Exception as e:
                logging.warning(f"Error reading file {structure_path}: {e}, skipping this structure.")
                continue

        return structures, labels

    def create_dataset(self, normalize=True) -> MGLDataset:
        """Create MGLDataset.

        Args:
            normalize (bool, optional): Whether to normalize. Defaults to True.

        Returns:
            MGLDataset: Created dataset object.
        """
        converter = Structure2Graph(element_types=self.element_list, cutoff=5.0)
        labels = {
            "energies": self.labels,
            "forces": [torch.zeros((len(s), 3)).tolist() for s in self.structures],
            "stresses": [torch.zeros((3, 3)).tolist() for _ in self.structures],
        }
        dataset = MGLDataset(
            structures=self.structures,
            labels=labels,
            converter=converter,
            include_line_graph=False  # Disable line graph to avoid unpacking errors
        )
        return dataset

    def create_dataloaders(self, dataset: MGLDataset):
        """Create training, validation, and test DataLoaders.

        Args:
            dataset (MGLDataset): Dataset object.

        Returns:
            Tuple[MGLDataLoader, MGLDataLoader, MGLDataLoader]: Training, validation, and test data loaders.
        """
        train_data, val_data, test_data = split_dataset(
            dataset=dataset,
            frac_list=self.split_ratio,
            shuffle=True,
            random_state=self.random_state
        )

        collate_fn = collate_fn_pes  # Use collate_fn_pes

        # Correctly call MGLDataLoader, passing in train_data, val_data, and test_data
        train_loader, val_loader, test_loader = MGLDataLoader(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            num_workers=0  # Adjust as needed
        )

        return train_loader, val_loader, test_loader


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
        Initialize the FineTuner for model fine-tuning.

        Args:
            working_dir (str): Directory where all outputs will be saved.
            pretrained_checkpoint (str): Path to the pretrained model checkpoint.
            freeze_base_layers (bool): Whether to freeze the base layers. Defaults to True.
            debug (bool): If True, sets logging level to DEBUG. Defaults to False.
            **kwargs: Additional keyword arguments for training configuration.
        """
        self.working_dir = Path(working_dir)
        self.debug = debug
        self.pretrained_checkpoint = pretrained_checkpoint
        self.freeze_base_layers = freeze_base_layers

        # Define directories for checkpoints, logs, and results
        self.checkpoints_dir = self.working_dir / "checkpoints"
        self.logs_dir = self.working_dir / "finetuning"
        self.results_dir = self.working_dir / "results"

        # Create directories if they do not exist
        for dir_path in [self.checkpoints_dir, self.logs_dir, self.results_dir]:
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
            'devices': kwargs.get('devices', 1),
            'split_ratio': kwargs.get('split_ratio', [0.8, 0.1, 0.1]),
            'random_state': kwargs.get('random_state', 42),
            'weight_decay': kwargs.get('weight_decay', 1e-5)
        }

        # Validate that the sum of split_ratio is 1.0
        if not abs(sum(self.config['split_ratio']) - 1.0) < 1e-6:
            raise ValueError("The sum of split_ratio must be 1.0")

        # Setup logging and save configuration
        self.setup_logging()
        self.save_config()

    def _cleanup_cache(self):
        """Clean up temporary files and cache."""
        cache_files = [
            "dgl_graph.bin",
            "lattice.pt",
            "dgl_line_graph.bin",
            "state_attr.pt",
            "labels.json",
            "M3GNet-MP-2021.2.8-PES"
        ]
        for fn in cache_files:
            file_path = Path(fn)
            if file_path.exists():
                file_path.unlink()

    def setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = self.logs_dir / 'train.log'
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)

        # Prevent adding handlers multiple times
        if not self.logger.handlers:
            # Create file handler
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG if self.debug else logging.INFO)

            # Create console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG if self.debug else logging.INFO)

            # Create formatter and add it to the handlers
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            # Add the handlers to the logger
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def save_config(self) -> None:
        """Save the training configuration to a JSON file."""
        config_path = self.results_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

    def setup_model(self, element_list) -> None:
        """Setup the M3GNet model and PyTorch Lightning trainer."""
        try:
            self.logger.info("Setting up M3GNet model...")
            # 1. Load the pretrained model
            self.model = matgl.load_model(self.pretrained_checkpoint)

            # Check if the model's element types match the data
            if hasattr(self.model.model, 'element_types'):
                pretrained_elements = set(self.model.model.element_types)
            else:
                self.logger.error("The model of the pretrained checkpoint lacks the 'element_types' attribute.")
                raise AttributeError("The model of the pretrained checkpoint lacks the 'element_types' attribute.")

            data_elements = set(element_list)
            if not data_elements.issubset(pretrained_elements):
                missing_elements = data_elements - pretrained_elements
                raise ValueError(f"The pretrained model does not contain the following elements: {missing_elements}")

            # 2. Freeze base layers if needed
            if self.freeze_base_layers:
                for name, param in self.model.model.named_parameters():
                    if "readout" not in name:
                        param.requires_grad = False
                self.logger.info("Base layers have been frozen.")

            # 3. Define the optimizer with different learning rates
            optimizer = torch.optim.Adam([
                {"params": [p for n, p in self.model.model.named_parameters() if "readout" not in n and p.requires_grad],
                 "lr": self.config['fine_tune_lr']},
                {"params": [p for n, p in self.model.model.named_parameters() if "readout" in n and p.requires_grad],
                 "lr": self.config['learning_rate']}
            ], weight_decay=self.config['weight_decay'])

            # 4. Initialize PotentialLightningModule and pass the custom optimizer
            self.lit_module = PotentialLightningModule(
                model=self.model,
                optimizer=optimizer,  # Pass the custom optimizer
                include_line_graph=False  # Ensure consistency with create_dataset
            )

            # 5. Configure the logger
            logger = CSVLogger(
                save_dir=str(self.logs_dir),
                name="",
                version=""
            )

            # 6. Initialize the PyTorch Lightning Trainer
            self.trainer = pl.Trainer(
                max_epochs=self.config['num_epochs'],
                accelerator=self.config['accelerator'],
                devices=self.config['devices'],
                logger=logger,
                log_every_n_steps=1,
                inference_mode=False  # Important to ensure gradients are computed during training
            )

            self.logger.info("Model setup completed.")

        except Exception as e:
            self.logger.error(f"Error setting up the model: {str(e)}")
            raise

    def run_training(self, paths: Dict) -> list:
        """Execute the model fine-tuning process."""
        start_time = time.time()
        self.logger.info("Starting fine-tuning process...")

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
            dataset = processor.create_dataset()

            # Check if the dataset is empty
            if len(dataset) == 0:
                raise ValueError("The dataset is empty. Please check the data files and labels.")

            # 3. Create DataLoaders
            train_loader, val_loader, test_loader = processor.create_dataloaders(dataset)

            # 4. Setup the model
            self.setup_model(processor.element_list)

            # 5. Start training
            self.logger.info("Starting training...")
            self.trainer.fit(
                model=self.lit_module,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader
            )

            # 6. Save the fine-tuned model
            model_save_path = self.checkpoints_dir / 'finetuned_model'
            self.lit_module.model.save(model_save_path)
            self.logger.info(f"Fine-tuned model saved to {model_save_path}")

            # 7. Evaluate the model on the test dataset
            test_results = self.trainer.test(
                model=self.lit_module,
                dataloaders=test_loader
            )

            # 8. Save test results
            results_file = self.results_dir / 'metrics.csv'
            pd.DataFrame(test_results).to_csv(results_file, index=False)

            # 9. Plot training curves (if metrics file exists)
            metrics_file = self.logs_dir / "metrics.csv"
            if metrics_file.exists():
                metrics = pd.read_csv(metrics_file)
                self.plot_training_curves(metrics)
            else:
                self.logger.warning(f"{metrics_file} not found, unable to plot training curves.")

            duration = time.time() - start_time
            self.logger.info(f"Fine-tuning completed in {duration:.2f} seconds.")
            return test_results

        except Exception as e:
            self.logger.error(f"Error during fine-tuning: {str(e)}")
            raise

        finally:
            self._cleanup_cache()

    def plot_training_curves(self, metrics: pd.DataFrame) -> None:
        """Plot and save the training and validation loss curves."""
        try:
            plt.figure(figsize=(10, 6))
            if "train_loss" in metrics.columns:
                metrics["train_loss"].dropna().plot(label='Training Loss')
            if "val_loss" in metrics.columns:
                metrics["val_loss"].dropna().plot(label='Validation Loss')

            plt.xlabel('Iterations')
            plt.ylabel('Loss')
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
            self.logger.info(f"Training curve saved to {plot_path}")

        except Exception as e:
            self.logger.error(f"Error plotting training curves: {str(e)}")


def main():
    # Get project paths
    paths = get_project_paths()

    # Ensure the output directory exists
    os.makedirs(paths['output_dir'], exist_ok=True)

    # Initialize FineTuner
    trainer = FineTuner(
        working_dir=paths['output_dir'],
        pretrained_checkpoint="M3GNet-MP-2021.2.8-PES",
        freeze_base_layers=True,
        num_epochs=10,
        learning_rate=1e-4,
        fine_tune_lr=1e-5,
        batch_size=32,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1  # Adjust according to your hardware
    )

    try:
        results = trainer.run_training(paths)
        print(f"Test results: {results}")
    finally:
        trainer._cleanup_cache()


if __name__ == "__main__":
    main()