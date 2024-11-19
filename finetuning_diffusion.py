from __future__ import annotations

import os
import warnings
import torch
import numpy as np
import lightning as pl
from torch import nn
import torch.nn.functional as F
from pytorch_lightning.loggers import CSVLogger
from matgl.models import M3GNet
from matgl.utils.training import ModelLightningModule
from dataset_process import process_dataset

class DiffusionModule(pl.LightningModule):
    def __init__(self, base_model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.base_model = base_model
        self.timesteps = timesteps
        
        # Define beta schedule
        self.beta = torch.linspace(beta_start, beta_end, timesteps)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def forward(self, x, t):
        return self.base_model(x, t)

    def training_step(self, batch, batch_idx):
        # Add noise and try to predict original data
        t = torch.randint(0, self.timesteps, (batch.shape[0],))
        noise = torch.randn_like(batch)
        noisy_batch = self.add_noise(batch, t, noise)
        predicted = self(noisy_batch, t)
        
        loss = F.mse_loss(predicted, noise)
        self.log('train_loss', loss)
        return loss

    def add_noise(self, x, t, noise):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise

def train_base_model(train_loader, val_loader, test_loader, elem_list, log_dir="logs/base_training"):
    model = M3GNet(
        element_types=elem_list,
        is_intensive=True,
        readout_type="set2set",
    )
    lit_module = ModelLightningModule(model=model)
    
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="cpu",
        logger=CSVLogger(log_dir)
    )
    
    trainer.fit(lit_module, train_loader, val_loader)
    trainer.test(lit_module, test_loader)
    
    model_path = os.path.join(log_dir, "checkpoints")
    lit_module.model.save(model_path)
    
    return lit_module, model_path

def fine_tune_model(pre_trained_path, train_loader, val_loader, elem_list, log_dir="logs/fine_tuned"):
    # Create a new instance of the M3GNet model
    model = M3GNet(
        element_types=elem_list,
        is_intensive=True,
        readout_type="set2set",
    )
    
    # Load the pre-trained model weights
    state_dict = torch.load(pre_trained_path)
    model.load_state_dict(state_dict)
    
    # Freeze early layers
    for param in model.embedding.parameters():
        param.requires_grad = False
        
    lit_module = ModelLightningModule(
        model=model,
        learning_rate=1e-4  # Lower learning rate for fine-tuning
    )
    
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="cpu",
        logger=CSVLogger(log_dir)
    )
    
    trainer.fit(lit_module, train_loader, val_loader)
    model_path = os.path.join(log_dir, "checkpoints")
    lit_module.model.save(model_path)
    
    return lit_module, model_path

def train_diffusion(fine_tuned_path, train_loader, val_loader, log_dir="logs/diffusion"):
    # Load the fine-tuned M3GNet model
    base_model = M3GNet(
        element_types=elem_list,
        is_intensive=True,
        readout_type="set2set",
    )
    state_dict = torch.load(fine_tuned_path)
    base_model.load_state_dict(state_dict)
    
    # Create the Diffusion Module
    diffusion = DiffusionModule(base_model)
    
    # Train the Diffusion Module
    trainer = pl.Trainer(
        max_epochs=15,
        accelerator="cpu",
        logger=CSVLogger(log_dir)
    )
    
    trainer.fit(diffusion, train_loader, val_loader)
    
    return diffusion

def generate_structures(diffusion_model, num_samples=5):
    with torch.no_grad():
        # Start from random noise
        x = torch.randn(num_samples, diffusion_model.base_model.hidden_dim)
        
        # Denoise step by step
        for t in reversed(range(diffusion_model.timesteps)):
            t_tensor = torch.tensor([t] * num_samples)
            x = diffusion_model.denoise_step(x, t_tensor)
            
    return x

if __name__ == "__main__":
    # Process dataset
    train_loader, val_loader, test_loader, elem_list = process_dataset()
    
    # Initial training
    base_module, base_path = train_base_model(train_loader, val_loader, test_loader, elem_list)
    
    # Fine-tuning
    fine_tuned_module, fine_tuned_path = fine_tune_model(base_path, train_loader, val_loader, elem_list)
    
    # Diffusion training
    diffusion_model = train_diffusion(fine_tuned_path, train_loader, val_loader)
    
    # Generate new structures
    new_structures = generate_structures(diffusion_model)