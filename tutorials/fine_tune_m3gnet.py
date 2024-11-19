import torch
import pytorch_lightning as pl
from matgl.models import M3GNet
from matgl.utils.training import PotentialLightningModule
from pytorch_lightning.loggers import CSVLogger
import matgl

class CustomPotentialLightningModule(PotentialLightningModule, pl.LightningModule):
    def __init__(self, model, lr, include_line_graph):
        super().__init__(model=model, lr=lr, include_line_graph=include_line_graph)
        self.model = model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        # Forward pass and compute loss
        loss = self.model.training_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # Forward pass and compute validation loss
        loss = self.model.validation_step(batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        # Define optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

def fine_tune_model(model_path="./finetuned_m3gnet_model/", pretrained_model_name="M3GNet-MP-2021.2.8-PES"):
    """Fine-tune a pre-trained M3GNet model."""
    # Load the pre-trained model
    pretrained_model = matgl.load_model(pretrained_model_name)
    
    # Initialize the LightningModule for fine-tuning
    lit_module_finetune = PotentialLightningModule(
        model=pretrained_model,
        lr=1e-4,
        include_line_graph=True
    )
    
    # Load DataLoaders
    data = torch.load('data_loaders.pth')
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    
    # Initialize the logger
    logger = CSVLogger("logs", name="M3GNet_finetuning")
    
    # Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=50,                # Adjust based on requirements
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        inference_mode=False,         # Required for training forces and stresses
    )
    
    # Start fine-tuning
    trainer.fit(
        model=lit_module_finetune,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    # Test the fine-tuned model
    trainer.test(dataloaders=data['test_loader'])
    
    # Save the fine-tuned model
    lit_module_finetune.model.save(model_path)
    print(f"Fine-tuned model saved to '{model_path}'.")

if __name__ == "__main__":
    fine_tune_model()
