from __future__ import annotations

import csv
import os
import warnings
import lightning as pl
import matplotlib.pyplot as plt
import pandas as pd
from pytorch_lightning.loggers import CSVLogger
from matgl.models import M3GNet
from matgl.utils.training import ModelLightningModule

warnings.simplefilter("ignore")

def train_model(train_loader, val_loader, test_loader, elem_list, log_dir="logs/M3GNet_training_small"):
    """Train M3GNet model for bandgap prediction with small dataset"""
    # Setup model
    model = M3GNet(
        element_types=elem_list,
        is_intensive=True,
        readout_type="set2set",
    )
    lit_module = ModelLightningModule(model=model)
    
    # Setup training
    logger = CSVLogger(log_dir)
    trainer = pl.Trainer(
        max_epochs=20,  # Reduced epochs for testing
        accelerator="cpu",
        logger=logger,
        num_sanity_val_steps=1,  # Reduced validation steps
        enable_checkpointing=True,
        enable_progress_bar=True
    )

    # Train and test
    trainer.fit(lit_module, train_loader, val_loader)
    test_results = trainer.test(lit_module, test_loader)
    
    # Save results
    working_dir = logger.log_dir
    model_export_path = os.path.join(working_dir, "checkpoints")
    lit_module.model.save(model_export_path)
    
    # Save metrics and plot training curve
    with open(os.path.join(working_dir, "test_metrics.csv"), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=test_results[0].keys())
        writer.writeheader()
        writer.writerows(test_results)
    
    metrics = pd.read_csv(os.path.join(working_dir, "metrics.csv"))
    plt.figure(figsize=(10, 6))
    metrics["train_MAE"].dropna().plot(label='Training')
    metrics["val_MAE"].dropna().plot(label='Validation')
    plt.xlabel('Iterations')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig(os.path.join(working_dir, "training_curve.png"))
    plt.close()
    
    return lit_module, working_dir

if __name__ == "__main__":
    # Import and process data
    from dataset_process import process_dataset
    train_loader, val_loader, test_loader, elem_list = process_dataset()
    
    # Train model
    lit_module, working_dir = train_model(train_loader, val_loader, test_loader, elem_list)
    
    print(f"Training completed. Results saved in {working_dir}")