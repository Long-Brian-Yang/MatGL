import os
import shutil
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matgl import load_model
from dataset_process import DataProcessor, get_project_paths
from training import BandgapTrainer

def finetune_m3gnet(pretrained_model_name: str, data_config: dict, trainer_config: dict, working_dir: Path) -> BandgapTrainer:
    """
    Fine-tune a pretrained M3GNet model.

    Args:
        pretrained_model_name (str): Name of pretrained M3GNet model (e.g. "M3GNet-MP-2021.2.8-PES")
        data_config (dict): Configuration dictionary for data processing
        trainer_config (dict): Configuration dictionary for trainer
        working_dir (Path): Working directory for saving outputs and models

    Returns:
        BandgapTrainer: Fine-tuned trainer instance
    """
    # Create working directory
    working_dir.mkdir(parents=True, exist_ok=True)
    print(f"Working directory set to: {working_dir}")

    # Download and load pretrained M3GNet model
    print(f"Downloading and loading pretrained model: {pretrained_model_name}")
    pretrained_m3gnet = load_model(pretrained_model_name)
    pretrained_model = pretrained_m3gnet.model
    print("Pretrained model loaded successfully.")

    # Process data
    print("Processing data...")
    processor = DataProcessor(data_config)
    processor.load_data()
    dataset = processor.create_dataset(normalize=True)
    train_loader, val_loader, test_loader = processor.create_dataloaders()
    print("Data processing completed.")

    # Initialize fine-tuning trainer
    print("Initializing BandgapTrainer...")
    finetune_trainer = BandgapTrainer(
        working_dir=str(working_dir),
        config=trainer_config
    )
    print("BandgapTrainer initialized.")

    # Initialize model with pretrained parameters
    print("Loading pretrained model parameters...")
    finetune_trainer.model = pretrained_model
    print("Pretrained parameters loaded.")

    # Fine-tune model
    print("Starting fine-tuning...")
    finetune_trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        element_types=processor.element_list
    )
    print("Fine-tuning completed.")

    # Evaluate fine-tuned model on test set
    print("Evaluating fine-tuned model...")
    results = finetune_trainer.evaluate(test_loader)
    print("Evaluation completed.")

    # Save fine-tuned model
    print("Saving fine-tuned model...")
    finetune_trainer.save_model(finetune_trainer.trainer.lightning_module)
    print("Model saved successfully.")

    # Plot training curves
    print("Plotting training curves...")
    finetune_trainer.plot_training_curves()
    print("Training curves saved.")

    # Save test results
    results_file = working_dir / 'test_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Test results saved to {results_file}")

    return finetune_trainer

def clean_up(working_dir: Path):
    """
    Clean up temporary files and directories.

    Args:
        working_dir (Path): Working directory path
    """
    temp_files = ["dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"]
    for fn in temp_files:
        try:
            os.remove(fn)
            print(f"Removed temporary file: {fn}")
        except FileNotFoundError:
            print(f"Temporary file not found, skipping: {fn}")

    dirs_to_remove = ["logs", working_dir / "checkpoints"]
    for dir_path in dirs_to_remove:
        try:
            shutil.rmtree(dir_path)
            print(f"Removed temporary directory: {dir_path}")
        except FileNotFoundError:
            print(f"Temporary directory not found, skipping: {dir_path}")

def main():
    """Main execution function for fine-tuning M3GNet model."""
    # Get project paths
    paths = get_project_paths()

    # Configure data processing
    data_config = {
        'structures_dir': paths['structures_dir'],  # Path to structure files directory
        'file_path': paths['file_path'],           # Path to data list file
        'cutoff': 4.0,
        'batch_size': 128
    }

    # Configure trainer
    trainer_config = {
        'batch_size': 128,
        'num_epochs': 2,
        'learning_rate': 1e-4,
        'accelerator': 'cpu'  # Change to 'gpu' if GPU is available
    }

    # Set working directory
    working_dir = Path(paths['output_dir']) / "M3GNet_finetuning"
    working_dir.mkdir(parents=True, exist_ok=True)
    print(f"Working directory set to: {working_dir}")

    # Pretrained model name
    pretrained_model_name = "M3GNet-MP-2021.2.8-PES"
    

    # Start fine-tuning
    finetune_trainer = finetune_m3gnet(
        pretrained_model_name=pretrained_model_name,
        data_config=data_config,
        trainer_config=trainer_config,
        working_dir=working_dir
    )

if __name__ == "__main__":
    main()