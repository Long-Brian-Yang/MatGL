from __future__ import annotations
import os
import json
import warnings
from pathlib import Path
from typing import Optional, Dict, Tuple
import csv

import numpy as np
import pandas as pd
import lightning as pl
import torch
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import CSVLogger
from matgl.models import M3GNet
from matgl.utils.training import ModelLightningModule
from dataset_process import DataProcessor, get_project_paths

warnings.simplefilter("ignore")


class BandgapTrainer:
    """
    使用 M3GNet 进行带隙预测的训练器。
    处理模型训练、验证和评估。
    """

    def __init__(
        self,
        working_dir: str,
        config: Optional[Dict] = None
    ):
        """
        使用配置初始化训练器。

        Args:
            working_dir (str): 输出的工作目录
            config (Optional[Dict]): 训练配置，包含：
                - batch_size (int): 批量大小
                - num_epochs (int): 训练的轮数
                - learning_rate (float): 学习率
                - cutoff_value (float): 截断半径
                - accelerator (str): 训练设备（'cpu' 或 'gpu'）
        """
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # 默认配置
        self.config = {
            'batch_size': 128,
            'num_epochs': 15,  # 从200改为15进行小规模训练
            'learning_rate': 0.001,
            'cutoff_value': 4.0,
            'accelerator': 'cpu',
            'random_state': 42,
            'save_top_k': 1,
            'early_stopping_patience': 5  # 为更快的早停调整
        }

        if config:
            self.config.update(config)

        # 初始化模型组件
        self.model = None
        self.trainer = None
        self.lit_module = None
        self.logger = None

        # 训练指标
        self.train_metrics = []
        self.val_metrics = []

        self.save_config()

    def save_config(self):
        """将训练配置保存到 JSON 文件中。"""
        config_path = self.working_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

    def setup_model(self, element_types: list):
        """
        设置 M3GNet 模型和训练组件。

        Args:
            element_types (list): 数据集中元素类型的列表
        """
        # 创建 M3GNet 模型
        self.model = M3GNet(
            element_types=element_types,
            is_intensive=True,
            readout_type="set2set"
        )

        # 创建 Lightning 模块
        self.lit_module = ModelLightningModule(
            model=self.model,
            lr=self.config['learning_rate']
        )

        # 设置日志记录器
        self.logger = CSVLogger(
            save_dir=str(self.working_dir),
            name="M3GNet_training"
        )

        # 创建 Trainer
        self.trainer = pl.Trainer(
            max_epochs=self.config['num_epochs'],
            accelerator=self.config['accelerator'],
            logger=self.logger
            # 由于删除了 callbacks，这里不再添加 ModelCheckpoint 和 EarlyStopping
        )

    def train(
        self, 
        train_loader, 
        val_loader,
        element_types: list
    ):
        """
        训练模型。

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            element_types (list): 元素类型列表
        """
        # 如果模型尚未设置，进行设置
        if self.model is None:
            self.setup_model(element_types)

        print(f"\n开始训练，共 {self.config['num_epochs']} 轮...")
        print(f"训练设备: {self.config['accelerator']}")

        # 训练模型
        self.trainer.fit(
            model=self.lit_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )

        # 保存模型
        self.save_model()

        # 绘制训练曲线
        self.plot_training_curves()

    def save_model(self):
        """保存训练好的模型。"""
        save_dir = self.working_dir / "checkpoints"
        save_dir.mkdir(exist_ok=True)

        # 获取最佳模型路径（由于移除了 ModelCheckpoint，此处可能无法获取）
        # 需要根据具体情况调整，例如手动保存最后一个模型
        last_model_path = self.trainer.checkpoint_callback.best_model_path if hasattr(self.trainer, 'checkpoint_callback') else None
        if last_model_path and os.path.exists(last_model_path):
            torch.save(self.lit_module.model.state_dict(), last_model_path)
            print(f"\n最佳模型已保存至 {last_model_path}")
        else:
            # 如果没有最佳模型路径，手动保存当前模型
            manual_save_path = save_dir / "model_last_epoch.pt"
            torch.save(self.lit_module.model.state_dict(), manual_save_path)
            print(f"\n模型已保存至 {manual_save_path}")

    def plot_training_curves(self):
        """绘制训练和验证曲线。"""
        metrics_path = self.working_dir / "M3GNet_training" / "metrics.csv"
        if not metrics_path.exists():
            print("未找到指标文件。跳过绘图。")
            return

        metrics = pd.read_csv(metrics_path)

        plt.figure(figsize=(10, 6))
        epochs = range(1, len(metrics["train_MAE"].dropna()) + 1)

        plt.plot(epochs, metrics["train_MAE"].dropna(), 
                 label='训练 MAE', marker='o')
        plt.plot(epochs, metrics["val_MAE"].dropna(), 
                 label='验证 MAE', marker='o')

        plt.xlabel('轮数')
        plt.ylabel('带隙 MAE')
        plt.title('M3GNet 训练进展')
        plt.legend()

        plt.savefig(
            self.working_dir / "training_curve.png",
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        print("\n训练曲线已保存。")

    def evaluate(self, test_loader) -> Dict:
        """
        在测试集上评估模型。

        Args:
            test_loader: 测试数据加载器

        Returns:
            dict: 评估指标
        """
        if self.trainer is None:
            raise ValueError("模型尚未训练。请先调用 train() 方法。")

        # 运行评估
        results = self.trainer.test(
            model=self.lit_module,
            dataloaders=test_loader
        )[0]

        print("\n测试结果:")
        print(f"MAE: {results.get('test_MAE', 'N/A'):.4f}")
        print(f"RMSE: {results.get('test_RMSE', 'N/A'):.4f}")

        # 保存结果
        results_file = self.working_dir / 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"\n测试结果已保存至 {results_file}")
        return results


def main():
    """BandgapTrainer 的示例用法。"""
    # 获取项目路径
    paths = get_project_paths()

    # 配置数据处理器
    data_config = {
        'structures_dir': paths['structures_dir'],
        'file_path': paths['file_path'],
        'cutoff': 4.0,
        'batch_size': 128
    }

    # 创建并处理数据
    processor = DataProcessor(data_config)
    processor.load_data()
    dataset = processor.create_dataset(normalize=True)
    train_loader, val_loader, test_loader = processor.create_dataloaders()

    # 保存数据拆分索引
    processor.save_split_indices('output/splits')

    # 配置训练器
    trainer_config = {
        'batch_size': 128,
        'num_epochs': 3,  # 设置为15轮进行小规模训练
        'learning_rate': 0.001,
        'accelerator': 'cpu',
        'save_top_k': 1,
        'early_stopping_patience': 5
    }

    # 创建并运行训练器
    trainer = BandgapTrainer(
        working_dir=Path(paths['output_dir']),
        config=trainer_config
    )

    # 训练模型
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        element_types=processor.element_list
    )

    # 评估模型
    results = trainer.evaluate(test_loader)

    print("\n训练已成功完成！")
    print(f"测试 MAE: {results.get('test_MAE', 'N/A'):.4f}")
    print(f"测试 RMSE: {results.get('test_RMSE', 'N/A'):.4f}")

    # 绘制并保存训练曲线
    working_dir = paths['output_dir'] # 请根据实际版本号调整
    metrics_path = os.path.join(working_dir, "metrics.csv")
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(metrics["train_MAE"].dropna()) + 1)

        plt.plot(epochs, metrics["train_MAE"].dropna(), 
                 label='训练 MAE', marker='o')
        plt.plot(epochs, metrics["val_MAE"].dropna(), 
                 label='验证 MAE', marker='o')

        plt.xlabel('轮数')
        plt.ylabel('带隙 MAE')
        plt.title('M3GNet 训练进展')
        plt.legend()

        plt.savefig(os.path.join(working_dir, "training_curve.png"), facecolor='w', bbox_inches="tight", pad_inches=0.3, transparent=True)
        plt.close()
        print("\n训练曲线已保存。")
    else:
        print("未找到指标文件。跳过绘图。")

    # 保存测试指标
    tests = trainer.trainer.test(model=trainer.lit_module, dataloaders=test_loader)
    filename = os.path.join(working_dir, "test_metrics.csv")
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=tests[0].keys())
        writer.writeheader()
        writer.writerows(tests)

    print(f"\n测试指标已保存至 {filename}")

    # 加载预训练模型并对新结构进行预测
    pretrained_model = M3GNet(
        element_types=processor.element_list,
        is_intensive=True,
        readout_type="set2set",
    )
    checkpoint_path = os.path.join(working_dir, "checkpoints", "model_last_epoch.pt")  # 请根据保存路径调整
    if os.path.exists(checkpoint_path):
        pretrained_model.load_state_dict(torch.load(checkpoint_path))
        pretrained_model.eval()  # 设置为评估模式
        print(f"\n已加载预训练模型来自 {checkpoint_path}")
    else:
        print(f"\n未找到预训练模型检查点 {checkpoint_path}")
        pretrained_model = None

    if pretrained_model:
        # 预测新结构
        test_file = '../../data/global_space/extracted_data/processed_data/Y8Ga4Zr4O24.vasp'
        test_struct = processor.read_poscar(test_file)
        bandgap = pretrained_model.predict_structure(structure=test_struct)
        print(f"预测的带隙为 {float(bandgap):.3f} eV。")
    else:
        print("无法进行预测，因为未加载预训练模型。")


if __name__ == "__main__":
    main()