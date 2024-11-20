"""
Diffusion coefficient analysis module.
Handles molecular dynamics simulation and diffusion analysis.
"""

from __future__ import annotations
import json
import warnings
from pathlib import Path
from typing import Optional, Dict

import torch
import numpy as np
from ase import Atoms, units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin

from dataset_process import DataProcessor, get_project_paths
from finetune import finetune_m3gnet

warnings.simplefilter("ignore")

class DiffusionAnalyzer:
    """Analyzer class for diffusion coefficient calculation via MD simulation."""
    
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
            'temperature_K': 1000,
            'timestep': 1.0,
            'maxtime_ps': 1.0,
            'friction': 0.01
        }
        if config:
            self.config.update(config)
        
        # Initialize directories
        (self.working_dir / "trajectories").mkdir(exist_ok=True)
        if self.debug:
            self.log_dir = self.working_dir / "debug_logs"
            self.log_dir.mkdir(exist_ok=True)
            
    def print_graph_info(self, graph):
        """Print graph feature information for debugging."""
        if not self.debug:
            return
            
        print("\nGraph Information:")
        print("Node features:", graph.ndata.keys())
        print("Edge features:", graph.edata.keys())
        for key in graph.ndata.keys():
            print(f"Feature '{key}' shape:", graph.ndata[key].shape)
            
    def graph_to_atoms(self, graph) -> Atoms:
        """
        Convert DGLGraph to ASE Atoms object.
        
        Args:
            graph: DGLGraph containing atomic structure information
        Returns:
            ase.Atoms: ASE Atoms object with atomic structure
        """
        if self.debug:
            self.print_graph_info(graph)
            
        # Get atomic positions
        position_keys = ['pos', 'coord', 'position', 'coordinates', 'R']
        positions = None
        for key in position_keys:
            if key in graph.ndata:
                positions = graph.ndata[key].numpy()
                break
        if positions is None:
            raise KeyError(f"Cannot find atomic positions. Available: {graph.ndata.keys()}")
        
        # Get atomic numbers
        type_keys = ['node_type', 'atom_types', 'Z', 'atomic_numbers']
        atomic_numbers = None
        for key in type_keys:
            if key in graph.ndata:
                atomic_numbers = graph.ndata[key].numpy()
                break
        if atomic_numbers is None:
            raise KeyError(f"Cannot find atomic numbers. Available: {graph.ndata.keys()}")
        
        # Get cell and pbc if available
        cell = None
        pbc = False
        cell_keys = ['lattice', 'cell', 'unit_cell']
        for key in cell_keys:
            if key in graph.ndata:
                cell = graph.ndata[key][0].numpy()
                pbc = [True, True, True]
                break
        
        return Atoms(
            numbers=atomic_numbers,
            positions=positions,
            cell=cell,
            pbc=pbc
        )
        
    def run_md_simulation(self, graph):
        """
        Run molecular dynamics simulation using graph data.
        
        Args:
            graph: DGLGraph object
        Returns:
            dict: Trajectory data including positions, energies, forces
        """
        # Convert graph to ASE Atoms
        structure = self.graph_to_atoms(graph)
        
        # Setup MD parameters
        temperature_K = self.config['temperature_K']
        timestep = self.config['timestep'] * units.fs
        maxtime_ps = self.config['maxtime_ps']
        friction = self.config['friction']
        
        # Initialize velocities
        MaxwellBoltzmannDistribution(structure, temperature_K=temperature_K)
        
        # Setup MD simulation
        trajectory_file = self.working_dir / "trajectories" / "temp.traj"
        dyn = Langevin(
            structure,
            timestep=timestep,
            temperature_K=temperature_K,
            friction=friction,
            trajectory=str(trajectory_file),
            logfile="md.log",
            loginterval=1
        )
        
        # Run simulation
        steps = int(maxtime_ps / (timestep * 1e-3))
        dyn.run(steps=steps)
        
        return self._process_trajectory(structure, trajectory_file)
        
    def _process_trajectory(self, structure, trajectory_file):
        """Process MD trajectory and extract data."""
        positions = []
        energies = []
        forces = []
        
        timestep = self.config['timestep'] * units.fs
        time_points = []
        current_time = 0
        
        for atoms in structure:
            positions.append(atoms.get_positions())
            energies.append(atoms.get_potential_energy())
            forces.append(atoms.get_forces())
            time_points.append(current_time)
            current_time += timestep
            
        return {
            'positions': torch.tensor(positions),
            'energies': torch.tensor(energies),
            'forces': torch.tensor(forces),
            'time': torch.tensor(time_points)
        }
        
    def calculate_msd(self, positions):
        """Calculate Mean Square Displacement."""
        disp = positions - positions[0]
        msd = torch.mean(torch.sum(disp**2, dim=2), dim=1)
        return msd
        
    def calculate_diffusion_coefficient(self, msd, time):
        """Calculate diffusion coefficient from MSD data."""
        slope = torch.polyfit(time, msd, 1)[0]
        D = slope / 6.0  # Einstein relation
        return D.item() * 1e-16 * 1e12  # Convert to cm²/s
        
    def analyze_diffusion(self, graph):
        """
        Analyze diffusion for a given structure.
        
        Args:
            graph: DGLGraph of structure
        Returns:
            float: Diffusion coefficient
        """
        # Run MD simulation
        traj_data = self.run_md_simulation(graph)
        
        # Calculate MSD and diffusion coefficient
        msd = self.calculate_msd(traj_data['positions'])
        D = self.calculate_diffusion_coefficient(msd, traj_data['time'])
        
        return D

def analyze_model_diffusion(trainer, test_loader):
    """
    Analyze diffusion coefficients for model predictions.
    
    Args:
        trainer: Trained model trainer
        test_loader: Test data loader
    Returns:
        dict: Analysis results
    """
    analyzer = DiffusionAnalyzer(
        working_dir=Path("diffusion_analysis")
    )
    
    predictions = []
    actual_values = []
    
    with torch.no_grad():
        for batch in test_loader:
            for graph in batch:
                try:
                    # Calculate actual diffusion coefficient
                    D_actual = analyzer.analyze_diffusion(graph)
                    
                    # Get model prediction
                    D_pred = trainer.model(graph)
                    
                    predictions.append(D_pred.item())
                    actual_values.append(D_actual)
                    
                except Exception as e:
                    if analyzer.debug:
                        print(f"Error analyzing sample: {str(e)}")
                    continue
    
    # Calculate metrics
    mae = np.mean(np.abs(np.array(predictions) - np.array(actual_values)))
    rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actual_values))**2))
    
    return {
        'test_MAE': mae,
        'test_RMSE': rmse,
        'predictions': predictions,
        'actual_values': actual_values
    }

def main():
    """Main execution function."""
    paths = get_project_paths()
    
    # Data configuration
    data_config = {
        'structures_dir': paths['structures_dir'],
        'file_path': paths['file_path'],
        'cutoff': 4.0,
        'batch_size': 128
    }
    
    # Trainer configuration
    trainer_config = {
        'batch_size': 128,
        'num_epochs': 2,
        'learning_rate': 1e-4,
        'accelerator': 'cpu'
    }
    
    # MD configuration
    md_config = {
        'temperature_K': 1000,
        'timestep': 1.0,
        'maxtime_ps': 1.0,
        'friction': 0.01
    }
    
    # Set working directory
    working_dir = Path(paths['output_dir']) / "diffusion_analysis"
    working_dir.mkdir(parents=True, exist_ok=True)
    
    # First fine-tune the model
    trainer = finetune_m3gnet(
        pretrained_model_name="M3GNet-MP-2021.2.8-PES",
        data_config=data_config,
        trainer_config=trainer_config,
        working_dir=working_dir
    )
    
    # Initialize data processor for test set
    processor = DataProcessor(data_config)
    processor.load_data()
    dataset = processor.create_dataset(normalize=True)
    _, _, test_loader = processor.create_dataloaders()
    
    # Analyze diffusion
    analyzer = DiffusionAnalyzer(
        working_dir=str(working_dir / "md_analysis"),
        config=md_config
    )
    
    # Calculate diffusion coefficients
    results = analyze_model_diffusion(trainer, test_loader)
    
    # Save results
    results_file = working_dir / 'diffusion_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Diffusion analysis results saved to {results_file}")

if __name__ == "__main__":
    main()