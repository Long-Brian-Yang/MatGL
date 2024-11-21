"""
Integrated molecular dynamics simulation and analysis module.
Provides comprehensive MD simulation capabilities and model comparison features.

Features:
- MD simulation with multiple thermostats
- Property calculations and analysis
- Model comparison visualization
- Comprehensive result logging
"""

from __future__ import annotations
import json
import warnings
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from ase import Atoms, units
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen
from matgl.models import M3GNet
from matgl import load_model

from dataset_process import DataProcessor, get_project_paths
from training import BandgapTrainer

class MDRunner:
    """
    Molecular dynamics simulation handler with comprehensive analysis tools.
    """
    
    def __init__(
        self,
        trainer: BandgapTrainer,
        working_dir: Path,
        config: Optional[Dict] = None,
        model_type: str = "finetuned"
    ):
        """
        Initialize MD simulation environment.
        
        Args:
            trainer: Trained model trainer instance
            working_dir: Working directory for outputs
            config: Optional configuration dictionary
            model_type: Type of model ("pretrained" or "finetuned")
        """
        self.trainer = trainer
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.model_type = model_type
        
        # Default configuration
        self.config = {
            'temperature_K': 300,
            'timestep_fs': 1.0,
            'equilibration_steps': 1000,
            'production_steps': 10000,
            'thermostat': 'langevin',
            'friction': 0.002,
            'taut': 0.1,
            'save_interval': 10,
            'random_seed': 42
        }
        if config:
            self.config.update(config)
            
        self._setup_directories()
        self.energy_log = []
        self.temperature_log = []
        
    # [Previous methods remain the same until calculate_diffusion_coefficient]
    
    def calculate_diffusion_coefficient(
        self,
        times: np.ndarray,
        msd: np.ndarray
    ) -> float:
        """
        Calculate diffusion coefficient from MSD data.
        
        Args:
            times: Time points in fs
            msd: MSD values in Å²
            
        Returns:
            Diffusion coefficient in cm²/s
        """
        times_ps = times * 1e-3
        slope, _ = np.polyfit(times_ps, msd, 1)
        D = slope / 6.0 * 1e-4
        return D

class ModelComparison:
    """
    Handles comparison between pre-trained and fine-tuned models.
    """
    
    def __init__(
        self,
        working_dir: Path,
        md_config: Optional[Dict] = None
    ):
        """
        Initialize model comparison handler.
        
        Args:
            working_dir: Working directory for outputs
            md_config: Optional MD configuration
        """
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.md_config = md_config or {}
        
    def setup_models(self) -> Tuple[MDRunner, MDRunner]:
        """
        Setup both pre-trained and fine-tuned model runners.
        
        Returns:
            Tuple of (pretrained_runner, finetuned_runner)
        """
        # Setup directories
        pretrained_dir = self.working_dir / "pretrained"
        finetuned_dir = self.working_dir / "finetuned"
        
        # Load pre-trained model
        pretrained_trainer = BandgapTrainer(
            working_dir=str(pretrained_dir),
            config={'batch_size': 128, 'num_epochs': 2}
        )
        pretrained_model = load_model("M3GNet-MP-2021.2.8-PES")
        pretrained_trainer.model = pretrained_model
        
        # Load fine-tuned model
        finetuned_trainer = BandgapTrainer(
            working_dir=str(finetuned_dir),
            config={'batch_size': 128, 'num_epochs': 2}
        )
        finetuned_trainer.load_model(finetuned_dir / "checkpoints" / "model.pt")
        
        # Create runners
        pretrained_runner = MDRunner(
            pretrained_trainer,
            pretrained_dir,
            self.md_config,
            "pretrained"
        )
        finetuned_runner = MDRunner(
            finetuned_trainer,
            finetuned_dir,
            self.md_config,
            "finetuned"
        )
        
        return pretrained_runner, finetuned_runner
        
    def run_comparison(self, structure: Atoms) -> Dict:
        """
        Run comparison analysis between models.
        
        Args:
            structure: Input structure for analysis
            
        Returns:
            Dictionary containing comparison results
        """
        pretrained_runner, finetuned_runner = self.setup_models()
        
        results = {}
        
        # Run analysis for pre-trained model
        print("\nRunning analysis with pre-trained model...")
        pretrained_data = pretrained_runner.run_production(structure)
        pretrained_msd = pretrained_runner.calculate_msd(pretrained_data['positions'])
        pretrained_D = pretrained_runner.calculate_diffusion_coefficient(
            pretrained_data['times'],
            pretrained_msd
        )
        
        # Run analysis for fine-tuned model
        print("\nRunning analysis with fine-tuned model...")
        finetuned_data = finetuned_runner.run_production(structure)
        finetuned_msd = finetuned_runner.calculate_msd(finetuned_data['positions'])
        finetuned_D = finetuned_runner.calculate_diffusion_coefficient(
            finetuned_data['times'],
            finetuned_msd
        )
        
        results = {
            'pretrained': {
                'times': pretrained_data['times'],
                'msd': pretrained_msd,
                'diffusion_coefficient': pretrained_D
            },
            'finetuned': {
                'times': finetuned_data['times'],
                'msd': finetuned_msd,
                'diffusion_coefficient': finetuned_D
            }
        }
        
        self.plot_comparison(results)
        self.save_results(results)
        
        return results
        
    def plot_comparison(self, results: Dict):
        """
        Create comparison plot between models.
        
        Args:
            results: Dictionary containing analysis results
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        
        # Convert times to ps
        times_ps_pre = results['pretrained']['times'] * 1e-3
        times_ps_fine = results['finetuned']['times'] * 1e-3
        
        # Pre-trained model plot
        ax1.plot(times_ps_pre, results['pretrained']['msd'], 'b-')
        ax1.set_xlabel('Time (ps)')
        ax1.set_ylabel('MSD (Angstrom²)')
        ax1.set_title('M3GNet, pre-trained')
        
        # Fine-tuned model plot
        ax2.plot(times_ps_fine, results['finetuned']['msd'], 'b-')
        ax2.set_xlabel('Time (ps)')
        ax2.set_ylabel('MSD (Angstrom²)')
        ax2.set_title('M3GNet, fine-tuned by VASP')
        
        plt.tight_layout()
        plt.savefig(
            self.working_dir / "model_comparison.png",
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        
    def save_results(self, results: Dict):
        """Save numerical results to JSON file."""
        save_data = {
            'pretrained_diffusion_coefficient': results['pretrained']['diffusion_coefficient'],
            'finetuned_diffusion_coefficient': results['finetuned']['diffusion_coefficient'],
            'md_configuration': self.md_config
        }
        
        with open(self.working_dir / "comparison_results.json", 'w') as f:
            json.dump(save_data, f, indent=4)
        
        print("\nComparison results:")
        print(f"Pre-trained D: {results['pretrained']['diffusion_coefficient']:.2e} cm²/s")
        print(f"Fine-tuned D: {results['finetuned']['diffusion_coefficient']:.2e} cm²/s")

def main():
    """Main execution function for model comparison."""
    paths = get_project_paths()
    
    # Load test structure
    structure = read(paths['structures_dir'] / "test_structure.vasp")
    
    # MD configuration
    md_config = {
        'temperature_K': 300,
        'timestep_fs': 1.0,
        'equilibration_steps': 1000,
        'production_steps': 10000,
        'thermostat': 'langevin',
        'save_interval': 10
    }
    
    # Setup working directory
    working_dir = Path(paths['output_dir']) / "model_comparison"
    
    # Run comparison
    comparison = ModelComparison(working_dir, md_config)
    results = comparison.run_comparison(structure)
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()