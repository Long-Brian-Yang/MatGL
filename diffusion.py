"""
Complete diffusion analysis module.
Handles molecular dynamics simulation and diffusion coefficient calculation.

Features:
- Graph to structure conversion
- MD simulation management
- Diffusion coefficient calculation
- Results visualization and analysis
"""

from __future__ import annotations
import json
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from ase import Atoms, units
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen
from matgl import load_model

from dataset_process import DataProcessor, get_project_paths
from training import BandgapTrainer

warnings.simplefilter("ignore")

class DiffusionAnalyzer:
    """
    Analyzer class for diffusion coefficient calculation via MD simulation.
    
    Handles:
    - Structure preparation
    - MD simulation
    - Diffusion analysis
    - Results visualization
    """
    
    def __init__(
        self, 
        working_dir: str,
        config: Optional[Dict] = None,
        debug: bool = True
    ):
        """
        Initialize diffusion analyzer.
        
        Args:
            working_dir: Directory for saving outputs
            config: Optional configuration dictionary
            debug: Enable debug output
        """
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        
        # Default MD configuration
        self.config = {
            'temperature_K': 300,
            'timestep_fs': 1.0,
            'maxtime_ps': 10.0,
            'friction': 0.002,
            'thermostat': 'langevin',
            'equilibration_steps': 1000,
            'production_steps': 10000,
            'save_interval': 10,
            'random_seed': 42
        }
        if config:
            self.config.update(config)
        
        # Initialize directories
        self._setup_directories()
        
        # Initialize loggers
        self.energy_log = []
        self.temperature_log = []
        
    def _setup_directories(self):
        """Create necessary output directories."""
        dirs = [
            'trajectories',
            'analysis',
            'debug_logs',
            'structures'
        ]
        for dir_name in dirs:
            (self.working_dir / dir_name).mkdir(exist_ok=True)
            
    def setup_md_simulation(self, atoms: Atoms) -> None:
        """
        Setup molecular dynamics simulation.
        
        Args:
            atoms: ASE Atoms object
        """
        # Set random seed
        np.random.seed(self.config['random_seed'])
        
        # Initialize velocities
        MaxwellBoltzmannDistribution(
            atoms, 
            temperature_K=self.config['temperature_K']
        )
        Stationary(atoms)  # Remove COM motion
        
        # Save initial structure
        write(
            self.working_dir / "structures" / "initial.traj",
            atoms
        )
        
    def run_md_simulation(self, atoms: Atoms) -> Dict:
        """
        Run molecular dynamics simulation.
        
        Args:
            atoms: ASE Atoms object
            
        Returns:
            Dict containing trajectory data
        """
        # Setup simulation
        self.setup_md_simulation(atoms)
        
        # Setup dynamics
        timestep = self.config['timestep_fs'] * units.fs
        
        if self.config['thermostat'] == 'langevin':
            dyn = Langevin(
                atoms,
                timestep=timestep,
                temperature_K=self.config['temperature_K'],
                friction=self.config['friction'],
                logfile=str(self.working_dir / "trajectories" / "md.log")
            )
        else:
            dyn = VelocityVerlet(
                atoms,
                timestep=timestep,
                logfile=str(self.working_dir / "trajectories" / "md.log")
            )
            
        # Run equilibration
        print("\nRunning equilibration...")
        for step in tqdm(range(self.config['equilibration_steps']), desc="Equilibration"):
            dyn.run(1)
            if step % self.config['save_interval'] == 0:
                self._log_properties(atoms)
                
        # Run production
        print("\nRunning production...")
        positions = []
        times = []
        current_time = 0.0
        
        for step in tqdm(range(self.config['production_steps']), desc="Production"):
            positions.append(atoms.get_positions())
            times.append(current_time)
            
            dyn.run(1)
            current_time += self.config['timestep_fs']
            
            if step % self.config['save_interval'] == 0:
                self._log_properties(atoms)
                
        # Save final structure
        write(
            self.working_dir / "structures" / "final.traj",
            atoms
        )
        
        return {
            'positions': np.array(positions),
            'times': np.array(times),
            'energy_log': np.array(self.energy_log),
            'temperature_log': np.array(self.temperature_log)
        }
        
    def _log_properties(self, atoms: Atoms) -> None:
        """Log system properties during simulation."""
        self.energy_log.append(atoms.get_total_energy())
        self.temperature_log.append(atoms.get_temperature())
        
    def calculate_msd(self, positions: np.ndarray) -> np.ndarray:
        """
        Calculate Mean Square Displacement.
        
        Args:
            positions: Array of positions [timesteps, atoms, 3]
            
        Returns:
            Array of MSD values
        """
        reference_pos = positions[0]
        msd = np.mean(np.sum((positions - reference_pos)**2, axis=2), axis=1)
        return msd
        
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
        # Convert times to ps for fitting
        times_ps = times * 1e-3
        
        # Fit line to MSD vs time
        slope, _ = np.polyfit(times_ps, msd, 1)
        
        # Convert to cm²/s (slope is in Å²/ps)
        D = slope / 6.0 * 1e-4
        return D
        
    def analyze_diffusion(self, atoms: Atoms) -> Tuple[float, Dict]:
        """
        Analyze diffusion for a given structure.
        
        Args:
            atoms: ASE Atoms object
            
        Returns:
            Tuple of (diffusion coefficient, analysis data)
        """
        # Run MD simulation
        traj_data = self.run_md_simulation(atoms)
        
        # Calculate MSD
        msd = self.calculate_msd(traj_data['positions'])
        
        # Calculate diffusion coefficient
        D = self.calculate_diffusion_coefficient(
            traj_data['times'],
            msd
        )
        
        # Save analysis data
        analysis_data = {
            'diffusion_coefficient': D,
            'msd': msd.tolist(),
            'times': traj_data['times'].tolist(),
            'energy': traj_data['energy_log'].tolist(),
            'temperature': traj_data['temperature_log'].tolist()
        }
        
        return D, analysis_data
        
    def plot_analysis(self, analysis_data: Dict):
        """
        Generate analysis plots.
        
        Args:
            analysis_data: Dictionary containing analysis results
        """
        # MSD plot
        times_ps = np.array(analysis_data['times']) * 1e-3
        msd = np.array(analysis_data['msd'])
        D = analysis_data['diffusion_coefficient']
        
        plt.figure(figsize=(10, 6))
        plt.plot(times_ps, msd, 'b-', label='MSD')
        plt.plot(times_ps, 6*D*times_ps*1e4, 'r--', 
                label=f'Fit (D = {D:.2e} cm²/s)')
        plt.xlabel('Time (ps)')
        plt.ylabel('MSD (Å²)')
        plt.title('Mean Square Displacement')
        plt.legend()
        plt.savefig(
            self.working_dir / "analysis" / "msd.png",
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        
        # Energy plot
        plt.figure(figsize=(10, 6))
        plt.plot(analysis_data['energy'])
        plt.xlabel('Step')
        plt.ylabel('Energy (eV)')
        plt.title('Energy Evolution')
        plt.savefig(
            self.working_dir / "analysis" / "energy.png",
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        
        # Temperature plot
        plt.figure(figsize=(10, 6))
        plt.plot(analysis_data['temperature'])
        plt.xlabel('Step')
        plt.ylabel('Temperature (K)')
        plt.title('Temperature Evolution')
        plt.savefig(
            self.working_dir / "analysis" / "temperature.png",
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

def analyze_model_diffusion(
    trainer: BandgapTrainer,
    test_loader,
    config: Optional[Dict] = None
) -> Dict:
    """
    Analyze diffusion coefficients using trained model.
    
    Args:
        trainer: Trained model trainer
        test_loader: Test data loader
        config: Optional MD configuration
        
    Returns:
        Dictionary containing analysis results
    """
    analyzer = DiffusionAnalyzer(
        working_dir="diffusion_analysis",
        config=config
    )
    
    results = {
        'predictions': [],
        'actual_values': [],
        'structures': []
    }
    
    successful_count = 0
    error_count = 0
    
    for batch in test_loader:
        for graph in batch:
            try:
                # Convert graph to atoms
                atoms = trainer.graph_to_atoms(graph)
                
                # Calculate actual diffusion coefficient
                D_actual, analysis_data = analyzer.analyze_diffusion(atoms)
                
                # Get model prediction
                with torch.no_grad():
                    D_pred = trainer.model(graph)
                
                results['predictions'].append(D_pred.item())
                results['actual_values'].append(D_actual)
                results['structures'].append(str(atoms))
                
                # Plot analysis for this structure
                analyzer.plot_analysis(analysis_data)
                
                successful_count += 1
                
            except Exception as e:
                print(f"Error analyzing structure: {str(e)}")
                error_count += 1
                continue
    
    print(f"\nAnalysis Summary:")
    print(f"Successfully analyzed: {successful_count} structures")
    print(f"Failed to analyze: {error_count} structures")
    
    if successful_count == 0:
        raise RuntimeError("No structures were successfully analyzed")
    
    # Calculate metrics
    predictions = np.array(results['predictions'])
    actual_values = np.array(results['actual_values'])
    
    mae = np.mean(np.abs(predictions - actual_values))
    rmse = np.sqrt(np.mean((predictions - actual_values)**2))
    
    results.update({
        'mae': mae,
        'rmse': rmse,
        'successful_count': successful_count,
        'error_count': error_count
    })
    
    return results

def main():
    """Main execution function."""
    try:
        paths = get_project_paths()
        
        # Data configuration
        data_config = {
            'structures_dir': paths['structures_dir'],
            'file_path': paths['file_path'],
            'cutoff': 4.0,
            'batch_size': 32
        }
        
        # MD configuration
        md_config = {
            'temperature_K': 300,
            'timestep_fs': 1.0,
            'maxtime_ps': 10.0,
            'friction': 0.002,
            'thermostat': 'langevin',
            'equilibration_steps': 1000,
            'production_steps': 10000
        }
        
        # Initialize data processor
        processor = DataProcessor(data_config)
        processor.load_data()
        dataset = processor.create_dataset(normalize=True)
        _, _, test_loader = processor.create_dataloaders()
        
        # Initialize trainer
        trainer = BandgapTrainer(
            working_dir=paths['output_dir'],
            config={'batch_size': 32}
        )
        
        # Run analysis
        print("\nStarting diffusion analysis...")
        results = analyze_model_diffusion(trainer, test_loader, md_config)
        
        # Save results
        output_dir = Path(paths['output_dir']) / "diffusion_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'diffusion_results.json', 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"\nResults saved to {output_dir / 'diffusion_results.json'}")
        print(f"MAE: {results['mae']:.4f}")
        print(f"RMSE: {results['rmse']:.4f}")
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()