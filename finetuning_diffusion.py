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
from matgl.utils.training import ModelLightningModule  # 改用 ModelLightningModule
from dataset_process import process_dataset

class DiffusionLightningModule(ModelLightningModule):
    """扩展 ModelLightningModule 来支持扩散"""
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__(model=model)
        self.timesteps = timesteps
        
        # 注册扩散参数
        self.register_buffer('beta', torch.linspace(beta_start, beta_end, timesteps))
        self.register_buffer('alpha', 1. - self.beta)
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))

    def training_step(self, batch, batch_idx):
        graph, targets = batch[0], batch[1]  # 假设batch[0]是图，batch[1]是目标值
        
        # 基础模型预测
        pred = self.model(graph)
        base_loss = F.mse_loss(pred, targets)
        
        # 添加扩散过程
        t = torch.randint(0, self.timesteps, (pred.shape[0],), device=self.device)
        noise = torch.randn_like(pred)
        noisy_pred = self.add_noise(pred, t, noise)
        predicted_noise = self.model(graph)  # 使用相同的图预测噪声
        
        # 计算扩散损失
        diffusion_loss = F.mse_loss(predicted_noise, noise)
        total_loss = base_loss + 0.1 * diffusion_loss
        
        # 记录损失
        self.log('train_base_loss', base_loss)
        self.log('train_diffusion_loss', diffusion_loss)
        self.log('train_total_loss', total_loss)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        graph, targets = batch[0], batch[1]
        pred = self.model(graph)
        loss = F.mse_loss(pred, targets)
        self.log('val_loss', loss)
        return loss

    def add_noise(self, x, t, noise):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise

def train_base_model(train_loader, val_loader, test_loader, elem_list, log_dir="logs/base_training"):
    """基础模型训练"""
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
    
    model_path = os.path.join(log_dir, "model.pt")
    torch.save(lit_module.model.state_dict(), model_path)
    
    return lit_module, model_path

def fine_tune_model(pre_trained_path, train_loader, val_loader, elem_list, log_dir="logs/fine_tuned"):
    """微调预训练模型"""
    model = M3GNet(
        element_types=elem_list,
        is_intensive=True,
        readout_type="set2set",
    )
    
    # 加载预训练权重
    state_dict = torch.load(pre_trained_path)
    model.load_state_dict(state_dict)
    
    # 冻结嵌入层
    for param in model.embedding.parameters():
        param.requires_grad = False
    
    # 使用扩散模块进行微调
    lit_module = DiffusionLightningModule(model=model)
    
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="cpu",
        logger=CSVLogger(log_dir)
    )
    
    trainer.fit(lit_module, train_loader, val_loader)
    model_path = os.path.join(log_dir, "model.pt")
    torch.save(lit_module.model.state_dict(), model_path)
    
    return lit_module, model_path

def generate_structures(diffusion_model, num_samples=5):
    """生成新结构"""
    with torch.no_grad():
        # 从随机噪声开始
        x = torch.randn(num_samples, diffusion_model.model.hidden_dim)
        
        # 逐步去噪
        for t in reversed(range(diffusion_model.timesteps)):
            t_tensor = torch.tensor([t] * num_samples, device=x.device)
            predicted_noise = diffusion_model.model(x)
            
            alpha = diffusion_model.alpha[t]
            alpha_bar = diffusion_model.alpha_bar[t]
            beta = diffusion_model.beta[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * predicted_noise) + \
                torch.sqrt(beta) * noise
            
    return x

if __name__ == "__main__":
    # 处理数据集
    train_loader, val_loader, test_loader, elem_list = process_dataset()
    
    # 基础训练
    base_module, base_path = train_base_model(train_loader, val_loader, test_loader, elem_list)
    
    # 微调并集成扩散
    fine_tuned_module, fine_tuned_path = fine_tune_model(base_path, train_loader, val_loader, elem_list)
    
    # 生成新结构
    new_structures = generate_structures(fine_tuned_module)