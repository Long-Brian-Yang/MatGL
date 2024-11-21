"""
Diffusion analysis module for crystal structures using fine-tuned M3GNet model.
"""

from __future__ import annotations
import os
import json
import warnings
from pathlib import Path
from typing import Optional, Dict, List

import torch
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms, units
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen
from matgl.ext.ase import PESCalculator
from matgl import load_model
from dataset_process import DataProcessor, get_project_paths
from training import BandgapTrainer

warnings.simplefilter("ignore")

class DiffusionAnalyzer:
    """Analyzer for diffusion coefficients via MD simulation using fine-tuned model."""
    
    def __init__(
        self, 
        working_dir: Path,
        model_path: str,  # Path to the fine-tuned model
        element_types: List[str],  # List of element types used in the model
        md_config: Optional[Dict] = None
    ):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_path = model_path
        self.element_types = element_types
        
        # Default MD configuration
        self.md_config = {
            'temperature_K': 300,
            'timestep_fs': 1.0,
            'friction': 0.002,
            'thermostat': 'langevin',
            'equilibration_steps': 1000,
            'production_steps': 10000
        }
        if md_config:
            self.md_config.update(md_config)
            
        # Load the fine-tuned model
        self.model = self.load_finetuned_model()
        
        # Create analysis directory
        (self.working_dir / "analysis").mkdir(exist_ok=True)
        
    def load_finetuned_model(self):
        """Load the fine-tuned model."""
        # Initialize BandgapTrainer without training
        trainer = BandgapTrainer(
            working_dir=str(self.working_dir)
        )
        # Setup model with element types
        trainer.setup_model(self.element_types)
        # Load fine-tuned weights
        trainer.model.load_state_dict(torch.load(self.model_path))
        # Set model to evaluation mode
        trainer.model.eval()
        return trainer.model
        
    def run_md(self, atoms: Atoms) -> Dict:
        """Run MD simulation using the fine-tuned model."""
        # Assign the model as the calculator
        atoms.calc = self.model
        
        # Set initial velocities
        MaxwellBoltzmannDistribution(
            atoms, 
            temperature_K=self.md_config['temperature_K']
        )
        
        # Setup dynamics
        if self.md_config['thermostat'] == 'langevin':
            dynamics = Langevin(
                atoms,
                timestep=self.md_config['timestep_fs'] * units.fs,
                temperature_K=self.md_config['temperature_K'],
                friction=self.md_config['friction']
            )
        else:
            dynamics = NVTBerendsen(
                atoms,
                timestep=self.md_config['timestep_fs'] * units.fs,
                temperature_K=self.md_config['temperature_K'],
                taut=100 * units.fs
            )
            
        # Data collection
        positions = []
        energies = []
        temperatures = []
        
        total_steps = self.md_config['equilibration_steps'] + self.md_config['production_steps']
        
        for step in range(total_steps):
            dynamics.run(1)
            if step >= self.md_config['equilibration_steps']:
                # Collect data during production phase
                positions.append(atoms.get_positions().copy())
            energies.append(atoms.get_total_energy())
            temperatures.append(atoms.get_temperature())
            
        return {
            'positions': positions,
            'energies': energies,
            'temperatures': temperatures
        }
        
    def analyze_diffusion(self, positions: List[np.ndarray]) -> float:
        """Calculate diffusion coefficient from trajectory positions."""
        # Calculate mean squared displacement (MSD)
        displacements = [pos - positions[0] for pos in positions]
        squared_displacements = [np.sum(d**2) for d in displacements]
        msd = np.mean(squared_displacements, axis=0)
        
        times = np.arange(len(positions)) * self.md_config['timestep_fs'] * 1e-15  # Convert fs to seconds
        
        # Linear fit to MSD vs time to get diffusion coefficient
        coeffs = np.polyfit(times, msd, 1)
        diffusion_coefficient = coeffs[0] / 6.0  # D = slope / 6 for 3D diffusion
        
        return diffusion_coefficient

def main():
    paths = get_project_paths()
    
    # Provide the path to the fine-tuned model
    model_path = os.path.join(paths['output_dir'], 'M3GNet_finetuning', 'checkpoints','model.pt')
    
    with open('element_types.json', 'r') as f:
        element_types = json.load(f)
    
    analyzer = DiffusionAnalyzer(
        working_dir=paths['diffusion_dir'],
        model_path=str(model_path),
        element_types=element_types
    )
    
    # Load initial structure
    structures_dir = Path(paths['structures_dir'])
    structure_files = list(structures_dir.glob('**/*.vasp'))
    
    for structure_file in structure_files:
        print(f"正在处理 {structure_file.name}")
        atoms = read(structure_file)

        structure_elements = set(atoms.get_chemical_symbols())
        missing_elements = structure_elements - set(element_types)

        # Run MD simulation
        md_results = analyzer.run_md(atoms)
    
        # Analyze diffusion coefficient
        diffusion_coefficient = analyzer.analyze_diffusion(md_results['positions'])
    
        print(f"Diffusion Coefficient: {diffusion_coefficient:.4e} m^2/s")
    
        # Save results
        results_file = analyzer.working_dir / 'diffusion_results.json'
        with open(results_file, 'w') as f:
            json.dump({'diffusion_coefficient': diffusion_coefficient}, f, indent=4)
    
        # Optional: Plot energies and temperatures
        plt.figure()
        plt.plot(md_results['energies'])
        plt.title('Total Energy')
        plt.savefig(analyzer.working_dir / 'analysis' / 'total_energy.png')
    
        plt.figure()
        plt.plot(md_results['temperatures'])
        plt.title('Temperature')
        plt.savefig(analyzer.working_dir / 'analysis' / 'temperature.png')

if __name__ == "__main__":
    main()