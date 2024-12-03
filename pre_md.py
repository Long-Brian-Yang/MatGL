from __future__ import annotations
import os
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from ase import Atoms
from ase.io import read, write, Trajectory
from ase.io.vasp import read_vasp
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import fs
from ase import Atom
import matgl
from matgl.ext.ase import PESCalculator
from pymatgen.io.ase import AseAtomsAdaptor
from dataset_process import get_project_paths, DataProcessor


class MDSystem:
    def __init__(
        self,
        config: dict,
        model_name: str = "M3GNet-MP-2021.2.8-PES",
        time_step: float = 1.0,  # fs
        friction: float = 0.02,
        total_steps: int = 20000,
        output_interval: int = 50
    ):
        # Get paths and setup directories
        self.paths = get_project_paths()
        self.structures_dir = Path(self.paths['structures_dir'])
        self.output_dir = Path(self.paths['output_dir'])
        self.md_output_dir = self.output_dir / 'md_trajectories'

        # Configuration
        self.config = config
        self.time_step = time_step
        self.friction = friction
        self.total_steps = total_steps
        self.output_interval = output_interval

        # Setup environment and logging
        self.setup_environment()

        # Initialize potential and calculator
        self.potential = matgl.load_model(model_name)
        self.calculator = PESCalculator(self.potential)
        self.logger.info(f"Loaded potential model: {model_name}")

        # Initialize data processor
        self.data_processor = DataProcessor(config)

        # Structure handlers
        self.atoms_adaptor = AseAtomsAdaptor()

    def setup_environment(self):
        """Setup logging and directories."""
        os.makedirs(self.md_output_dir, exist_ok=True)

        # Setup logging
        log_file = self.md_output_dir / \
            f"md_simulation_{datetime.now():%Y%m%d_%H%M%S}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("main")
        self.logger.info(f"Working directory: {self.md_output_dir}")
        self.logger.info(f"Structures directory: {self.structures_dir}")

    def find_vasp_files(self) -> List[Path]:
        """Find all .vasp files in structures directory."""
        vasp_files = list(self.structures_dir.glob("**/*.vasp"))
        self.logger.info(f"Found {len(vasp_files)} .vasp files")
        for f in vasp_files:
            self.logger.info(f"Found structure file: {f}")
        return vasp_files

    def add_protons(self, atoms: Atoms, n_protons: int) -> Atoms:
        """
        Add protons to the structure near oxygen atoms.

        Args:
            atoms (Atoms): Input structure
            n_protons (int): Number of protons to add

        Returns:
            Atoms: Structure with protons added
        """
        # Theoretical OH bond length (from Gomez et al.)
        OH_BOND_LENGTH = 0.98  # Å
        MAX_NEIGHBOR_DIST = 3.0  # Å for neighbor search

        # Find oxygen atoms
        o_indices = [i for i, symbol in enumerate(atoms.get_chemical_symbols())
                     if symbol == 'O']

        if len(o_indices) < n_protons:
            self.logger.warning(
                f"Number of protons ({n_protons}) exceeds number of O atoms ({len(o_indices)})")
            n_protons = len(o_indices)

        # Track used oxygen atoms
        used_oxygens = []

        # Get cell and PBC
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()

        # Add protons near selected oxygen atoms
        for i in range(n_protons):
            # Select oxygen atom that hasn't been used
            available_oxygens = [
                idx for idx in o_indices if idx not in used_oxygens]
            if not available_oxygens:
                self.logger.warning(
                    "No more available oxygen atoms for proton incorporation")
                break

            o_idx = available_oxygens[0]
            used_oxygens.append(o_idx)
            o_pos = atoms.positions[o_idx]

            # Find neighboring oxygen atoms to determine optimal proton
            neighbors = []
            for other_idx in o_indices:
                if other_idx != o_idx:
                    dist = atoms.get_distance(o_idx, other_idx, mic=True)
                    if dist < MAX_NEIGHBOR_DIST:
                        vec = atoms.get_distance(
                            o_idx, other_idx, vector=True, mic=True)
                        neighbors.append({
                            'idx': other_idx,
                            'dist': dist,
                            'vec': vec
                        })

            # Calculate optimal proton position
            direction = np.zeros(3)
            if neighbors:
                for n in sorted(neighbors, key=lambda x: x['dist'])[:3]:
                    weight = 1.0 / max(n['dist'], 0.1)
                    direction -= n['vec'] * weight

                if np.linalg.norm(direction) > 1e-6:
                    direction = direction / np.linalg.norm(direction)
                else:
                    direction = np.array([0, 0, 1])
            else:
                direction = np.array([0, 0, 1])

            # Calculate proton position
            h_pos = o_pos + direction * OH_BOND_LENGTH

            min_allowed_dist = 0.8  # Å
            for pos in atoms.positions:
                dist = atoms.get_distance(-1, len(atoms) - 1, mic=True)
                if dist < min_allowed_dist:
                    is_valid = False
                    break
            if any(pbc):
                scaled_pos = np.linalg.solve(cell.T, h_pos.T).T
                scaled_pos = scaled_pos % 1.0
                h_pos = cell.T @ scaled_pos

            is_valid = True
            for pos in atoms.positions:
                dist = np.linalg.norm(h_pos - pos)
                if dist < 0.5:
                    is_valid = False
                    break

            if not is_valid:
                self.logger.warning(
                    f"Invalid proton position near O atom {o_idx}, trying different direction")
                continue

            # Add proton
            atoms.append(Atom('H', position=h_pos))

            # Log OH bond length for verification
            oh_dist = atoms.get_distance(-1, o_idx, mic=True)
            self.logger.info(f"Added proton {i+1}/{n_protons}:")
            self.logger.info(f"  Near O atom: {o_idx}")
            self.logger.info(f"  Position: {h_pos}")
            self.logger.info(f"  OH distance: {oh_dist:.3f} Å")

        self.logger.info(f"Successfully added {n_protons} protons")
        self.logger.info(f"Final composition: {atoms.get_chemical_formula()}")

        return atoms

    def run_md(
        self,
        structure_file: Path,
        n_protons: int,
        temperatures: List[float]
    ) -> Dict[float, str]:
        """
        Run molecular dynamics simulation for a structure with protons at multiple temperatures.

        Args:
            structure_file (Path): Path to the structure file
            n_protons (int): Number of protons to add
            temperatures (List[float]): List of temperatures (K) for simulation

        Returns:
            Dict[float, str]: Dictionary mapping temperatures to trajectory file paths
        """
        # Create output directory for this material
        material_dir = self.md_output_dir / \
            f"{structure_file.stem}_H{n_protons}"
        os.makedirs(material_dir, exist_ok=True)

        try:
            # Prepare hydrated structure
            atoms = read_vasp(str(structure_file))
            atoms = self.add_protons(atoms, n_protons)
            hydrated_file = material_dir / \
                f"{structure_file.stem}_H{n_protons}.vasp"
            write(str(hydrated_file), atoms, format='vasp')
            self.logger.info(
                f"Prepared hydrated structure: {atoms.get_chemical_formula()}")

            # Add calculator
            atoms.calc = self.calculator

            # Run MD at each temperature
            trajectories = {}
            for temp in temperatures:
                temp_dir = material_dir / f"T_{temp}K"
                os.makedirs(temp_dir, exist_ok=True)

                try:
                    # Setup trajectory file
                    traj_file = temp_dir / f"MD_{int(temp)}K.traj"
                    traj = Trajectory(str(traj_file), 'w', atoms)

                    # Initialize velocities for this temperature
                    MaxwellBoltzmannDistribution(atoms, temperature_K=temp)

                    # Setup Langevin dynamics
                    dyn = Langevin(
                        atoms,
                        timestep=self.time_step * fs,
                        temperature_K=temp,
                        friction=self.friction
                    )
                    dyn.attach(traj.write, interval=self.output_interval)

                    # Run MD
                    self.logger.info(f"Starting MD at {temp}K")
                    self.logger.info(
                        f"Total steps: {self.total_steps}, Output interval: {self.output_interval}")

                    for step in range(1, self.total_steps + 1):
                        dyn.run(1)
                        if step % 1000 == 0:
                            current_temp = atoms.get_temperature()
                            self.logger.info(
                                f"Step {step}/{self.total_steps}, Temperature: {current_temp:.1f}K")

                            # Save checkpoint
                            checkpoint_file = temp_dir / \
                                f"POSCAR_checkpoint_{step}"
                            write(str(checkpoint_file), atoms, format='vasp')

                    # Save final structure
                    final_structure = temp_dir / f"FINAL_POSCAR_{int(temp)}K"
                    write(str(final_structure), atoms, format='vasp')

                    trajectories[temp] = str(traj_file)
                    self.logger.info(
                        f"Completed MD at {temp}K, saved final structure to {final_structure}")

                except Exception as e:
                    self.logger.error(f"Error running MD at {temp}K: {str(e)}")
                    continue

            return trajectories

        except Exception as e:
            self.logger.error(f"Error in MD simulation setup: {str(e)}")
            raise


def main():
    """Main function to run MD simulation."""
    try:
        # Setup parameters
        paths = get_project_paths()
        simulation_params = {
            'data_config': {
                'structures_dir': paths['structures_dir'],
                'file_path': paths['file_path'],
                'cutoff': 4.0,
                'batch_size': 16,
                'split_ratio': [0.7, 0.1, 0.2],
                'random_state': 42
            },
            'md_params': {
                'model_name': "M3GNet-MP-2021.2.8-PES",
                'time_step': 1.0,
                'friction': 0.02,
                'total_steps': 10000,
                'output_interval': 50
            },
            'simulation': {
                'structure_file': "data/Ba8Zr8O24.vasp",
                'n_protons': 2,
                'temperatures': [600]  # K
            }
        }

        # Initialize MD system
        md_system = MDSystem(
            config=simulation_params['data_config'],
            **simulation_params['md_params']
        )

        # Get simulation parameters
        structure_file = Path(
            simulation_params['simulation']['structure_file'])
        n_protons = simulation_params['simulation']['n_protons']
        temperatures = simulation_params['simulation']['temperatures']

        # Run simulation using the class method
        trajectories = md_system.run_md(
            structure_file, n_protons, temperatures)

        # Save configuration
        simulation_params['results'] = {
            'structure_file': str(structure_file),
            'trajectories': trajectories
        }
        config_file = md_system.md_output_dir / 'md_config.json'
        with open(config_file, 'w') as f:
            json.dump(simulation_params, f, indent=4)

        print("\nSimulation completed successfully!")
        print(f"Results saved to: {config_file}")

    except Exception as e:
        print(f"\nError in MD simulation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
