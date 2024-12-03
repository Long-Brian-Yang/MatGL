from __future__ import annotations
import os
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from ase import Atoms
from ase.io import read, write, Trajectory
from ase.io.vasp import read_vasp
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import fs
from ase import Atom
from ase.calculators.calculator import Calculator
import torch
import matgl
from matgl.ext.ase import PESCalculator
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from dataset_process import get_project_paths, DataProcessor


class MDSystem:
    def __init__(
        self,
        config: dict,
        model_checkpoint: str,  # Changed from model_name to model_checkpoint
        time_step: float = 1.0,
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

        # Initialize fine-tuned model and calculator
        self.load_finetuned_model(model_checkpoint)

        # Initialize data processor
        self.data_processor = DataProcessor(config)

        # Structure handlers
        self.atoms_adaptor = AseAtomsAdaptor()

    def load_finetuned_model(self, checkpoint_path: str):
        """Load the fine-tuned M3GNet model."""
        try:
            # Load the fine-tuned model
            self.potential = torch.load(checkpoint_path)
            self.calculator = PESCalculator(self.potential)
            self.logger.info(
                f"Loaded fine-tuned model from: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Error loading fine-tuned model: {str(e)}")
            raise

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
        if not vasp_files:
            self.logger.warning(
                f"No .vasp files found in {self.structures_dir}")
        else:
            self.logger.info(f"Found {len(vasp_files)} .vasp files")
            for f in vasp_files:
                self.logger.info(f"Found structure file: {f}")
        return vasp_files

    def add_protons(self, atoms: Atoms, n_protons: int) -> Atoms:
        """
        Add protons to the structure based on theoretical understanding.

        Args:
            atoms (Atoms): Initial structure
            n_protons (int): Number of protons to add

        Returns:
            Atoms: Structure with protons added

        References:
        1. Kreuer, K. D., Solid State Ionics (1999)
        2. Björketun, M. E., et al., PRB (2005)
        3. Gomez, M. A., et al., SSI (2010)
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
        temperature: float,
        traj_file: Optional[Path] = None
    ) -> str:
        """
        Run MD simulation for given structure and temperature.
        """
        # Create output directory for this structure
        struct_name = structure_file.stem
        struct_output_dir = self.md_output_dir / struct_name
        os.makedirs(struct_output_dir, exist_ok=True)

        # Read structure using ASE's VASP reader
        try:
            atoms = read_vasp(str(structure_file))
            atoms.calc = self.calculator

            self.logger.info(f"Loaded structure from {structure_file}")
            self.logger.info(
                f"Structure composition: {atoms.get_chemical_formula()}")

        except Exception as e:
            self.logger.error(
                f"Error loading structure from {structure_file}: {e}")
            raise

        # Initialize velocities
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

        # Setup Langevin dynamics
        dyn = Langevin(
            atoms,
            timestep=self.time_step * fs,
            temperature_K=temperature,
            friction=self.friction
        )

        # Setup trajectory file
        if traj_file is None:
            traj_file = struct_output_dir / f"MD_{int(temperature):04d}.traj"
        traj = Trajectory(str(traj_file), 'w', atoms)
        dyn.attach(traj.write, interval=self.output_interval)

        # Run MD
        self.logger.info(f"Starting MD at {temperature}K for {struct_name}")
        self.logger.info(
            f"Total steps: {self.total_steps}, Output interval: {self.output_interval}")

        for step in range(1, self.total_steps + 1):
            dyn.run(1)
            if step % 1000 == 0:
                temp = atoms.get_temperature()
                self.logger.info(
                    f"Step {step}/{self.total_steps}, Temperature: {temp:.1f}K")

                # Save current state as VASP format
                checkpoint_file = struct_output_dir / \
                    f"POSCAR_checkpoint_{step}"
                write(str(checkpoint_file), atoms, format='vasp')

        self.logger.info(
            f"MD simulation completed. Trajectory saved to {traj_file}")

        return str(traj_file)

    def run_temperature_range(
        self,
        structure_files: List[Path] = None,
        temperatures: List[float] = None
    ) -> Dict[str, Dict[float, str]]:
        """Run MD simulations for multiple structures at different temperatures."""
        if structure_files is None:
            structure_files = self.find_vasp_files()

        if not structure_files:
            self.logger.error("No structure files found")
            return {}

        if temperatures is None:
            temperatures = [500, 700, 1000]  # default temperatures

        results = {}
        for struct_file in structure_files:
            struct_name = struct_file.stem
            results[struct_name] = {}

            for temp in temperatures:
                try:
                    traj_file = self.run_md(struct_file, temp)
                    results[struct_name][temp] = traj_file
                except Exception as e:
                    self.logger.error(
                        f"Error running MD for {struct_name} at {temp}K: {str(e)}")

        return results

# def main():
#     """Main function to run MD simulations."""
#     # Get project paths
#     paths = get_project_paths()

#     # Setup configuration
#     config = {
#         'structures_dir': paths['structures_dir'],
#         'file_path': paths['file_path'],
#         'cutoff': 4.0,
#         'batch_size': 16,
#         'split_ratio': [0.5, 0.1, 0.4],
#         'random_state': 42
#     }

#     # Initialize MD system
#     md_system = MDSystem(
#         config=config,
#         time_step=1.0,
#         friction=0.02,
#         total_steps=20000,
#         output_interval=100
#     )

#     # Define temperatures
#     temperatures = [800, 900, 1000]  # K

#     # Run MD simulations for all .vasp files
#     print("\nRunning MD simulations...")
#     results = md_system.run_temperature_range(temperatures=temperatures)

#     print("\nCompleted MD simulations:")
#     for struct_name, temp_dict in results.items():
#         print(f"\nStructure: {struct_name}")
#         for temp, traj_file in temp_dict.items():
#             print(f"  {temp}K: {traj_file}")

# if __name__ == "__main__":
#     main()


def main():
    """Test MD simulation with fine-tuned model."""
    paths = get_project_paths()

    # Setup configuration
    config = {
        'structures_dir': paths['structures_dir'],
        'file_path': paths['file_path'],
        'cutoff': 4.0,
        'batch_size': 16,
        'split_ratio': [0.7, 0.1, 0.2],
        'random_state': 42
    }

    # Path to your fine-tuned model checkpoint
    finetuned_checkpoint = os.path.join(paths['output_dir'], "checkpoints")

    # Initialize MD system with fine-tuned model
    md_system = MDSystem(
        config=config,
        model_checkpoint=finetuned_checkpoint,  # Use fine-tuned model
        time_step=1.0,
        friction=0.02,
        total_steps=20000,
        output_interval=50
    )

    # Test with specific structure
    structure_file = Path("data/Ba8Zr8O24.vasp")
    print(f"\nTesting with structure: {structure_file}")

    try:
        # 1. Create material directory
        n_protons = 1
        material_dir = md_system.md_output_dir / \
            f"{structure_file.stem}_H{n_protons}"
        os.makedirs(material_dir, exist_ok=True)

        # 2. Load initial structure
        atoms = read_vasp(str(structure_file))
        print(f"\nLoaded initial structure: {atoms.get_chemical_formula()}")

        # 3. Add protons
        atoms = md_system.add_protons(atoms, n_protons)

        # 4. Save hydrated structure
        hydrated_file = md_system.md_output_dir / \
            f"{structure_file.stem}_H{n_protons}.vasp"
        write(str(hydrated_file), atoms, format='vasp')
        print(f"Saved hydrated structure to: {hydrated_file}")

        # 5. Run MD simulations
        temperatures = [800]  # K
        trajectories = {}

        for temp in temperatures:
            print(f"\nRunning MD at {temp}K...")

            # Create temperature directory
            temp_dir = material_dir / f"T_{temp}K"
            os.makedirs(temp_dir, exist_ok=True)

            try:
                # Run MD simulation
                traj_path = md_system.run_md(
                    structure_file=hydrated_file,
                    temperature=temp,
                    traj_file=temp_dir / f"MD_{temp}K.traj"
                )

                trajectories[temp] = traj_path
                print(f"Completed MD at {temp}K")
                print(f"Trajectory saved to: {traj_path}")

                # Save final structure
                final_structure = temp_dir / f"FINAL_POSCAR_{temp}K"
                final_atoms = read(traj_path, index=-1)
                write(str(final_structure), final_atoms, format='vasp')
                print(f"Final structure saved to: {final_structure}")

            except Exception as e:
                print(f"Error running MD at {temp}K: {str(e)}")
                continue

        print("\nMD Simulation Summary:")
        print("-" * 50)
        print(f"Initial structure: {structure_file}")
        print(f"Protons added: {n_protons}")
        print(f"Temperatures simulated: {temperatures} K")
        print("\nTrajectory files:")
        for temp, traj in trajectories.items():
            print(f"  {temp}K: {traj}")

    except Exception as e:
        print(f"\nError in MD simulation: {str(e)}")
        raise

    finally:
        config_info = {
            'structure_file': str(structure_file),
            'n_protons': n_protons,
            'temperatures': temperatures,
            'md_parameters': {
                'time_step': md_system.time_step,
                'friction': md_system.friction,
                'total_steps': md_system.total_steps,
                'output_interval': md_system.output_interval
            }
        }

        config_file = md_system.md_output_dir / 'md_config.json'
        with open(config_file, 'w') as f:
            json.dump(config_info, f, indent=4)
        print(f"\nConfiguration saved to: {config_file}")


if __name__ == "__main__":
    main()
