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
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from ase import Atoms, units
from ase.io import read, write, Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import J, mol
from matplotlib.animation import FuncAnimation
from distutils.util import strtobool
import argparse
import ast
from datetime import datetime
import logging

from matgl.ext.ase import PESCalculator
from matgl import load_model
from dataset_process import DataProcessor, get_project_paths
from training import BandgapTrainer

warnings.simplefilter("ignore")

# Configure matplotlib
mpl.use('Agg')  # Use a non-interactive backend
plt.rcParams.update({
    'axes.labelsize': 26,
    'font.size': 26,
    'font.family': 'DejaVu Sans',
    'legend.fontsize': 26,
    'xtick.labelsize': 26,
    'ytick.labelsize': 26,
    'axes.titlesize': 26,
    'text.usetex': False,
    'figure.figsize': [14, 14]
})

font_size = 26

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
            'equilibration_steps': 500,
            'production_steps': 1000
        }
        if md_config:
            self.md_config.update(md_config)
            
        # Load the fine-tuned model
        self.potential = self.load_finetuned_model()

        self.calculator = PESCalculator(self.potential)

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
        # Assign custom calculator
        atoms.calc = self.calculator

        # Set initial velocities
        MaxwellBoltzmannDistribution(atoms, temperature_K=self.md_config['temperature_K'])

        # Setup dynamics
        if self.md_config['thermostat'] == 'langevin':
            dynamics = Langevin(
                atoms,
                timestep=self.md_config['timestep_fs'] * units.fs,
                temperature_K=self.md_config['temperature_K'],
                friction=self.md_config['friction'],
            )
        else:
            raise ValueError(f"Unsupported thermostat: {self.md_config['thermostat']}")

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
            energies.append(atoms.get_potential_energy())
            temperatures.append(atoms.get_temperature())

        return {
            'positions': positions,
            'energies': energies,
            'temperatures': temperatures,
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

# Additional Analysis and Plotting Functions

def volume_cal_f(POSCAR_file):
    with open(POSCAR_file, 'r') as f:
        lines = f.readlines()
    
    # Scaling factor
    scaling_factor = float(lines[1].strip())
    
    # Lattice vectors
    lattice_vectors = []
    for i in range(2, 5):
        vector = list(map(float, lines[i].split()))
        lattice_vectors.append(vector)
    
    lattice_vectors = np.array(lattice_vectors) * scaling_factor
    volume = np.abs(np.linalg.det(lattice_vectors))
    # Convert volume from Å^3 to cm^3 (1 Å = 1e-8 cm)
    volume_cm3 = volume * 1e-24
    return volume_cm3

def proton_conductivity_cal_f(number_of_protons, diffusion_D, volume_sys, temp):
    # Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7761793/
    # Constants
    e = 1.60217662e-19  # Elementary charge in Coulombs
    k_B = 1.38064852e-23  # Boltzmann constant in J/K
    T = temp  # Temperature in Kelvin
    volume = volume_sys  # Volume in cubic meters
    n_protons = number_of_protons  # Number of protons
    sigma = (n_protons * e**2 * diffusion_D) / (volume * k_B * T)
    return sigma

def MSD_D_calculation_f(working_MD_dir, T_list):
    atom_diffusion_lbl = 'H'
    shift_t = 500
    window_size = 1000
    step = 1
    D_all = []
    for T in T_list:
        traj_file = os.path.join(working_MD_dir, f"MD_{T}.traj")
        if not os.path.isfile(traj_file):
            print(f'{traj_file} does not exist!')
            continue
        traj_list = read(traj_file, index=":")
        atom_index = [i for i, x in enumerate(traj_list[0].get_chemical_symbols()) if x == atom_diffusion_lbl]
        volume = [atoms.get_volume() for atoms in traj_list]
        print(f"Index of {atom_diffusion_lbl} atom: {atom_index}")
        positions_all = np.array([traj_list[i].get_positions() for i in range(len(traj_list))])
        positions = positions_all[:, atom_index]
        msd = np.mean(np.sum((positions - positions[0])**2, axis=2), axis=1)
        msd_list = []
        D_list = []
        slope_list = []
        intercept_list = []
        for i in range(0, int(len(msd)/shift_t)):
            msd_t = np.mean(np.sum((positions[i*shift_t:i*shift_t + window_size] - positions[i*shift_t])**2, axis=2), axis=1)
            if len(msd_t) != window_size:
                continue
            slope, intercept = np.polyfit(range(0, window_size, step), msd_t[::step], 1)
            D = slope / 6
            D_list.append(D)
            slope_list.append(slope)
            intercept_list.append(intercept)
            msd_list.append(msd_t[::step])
        ave_D = np.mean(D_list)
        D_cm = ave_D * 1e-16 / 1e-12
        log10_D_cm = np.log10(D_cm)
        number_of_proton = len(positions)
        volume_cm3 = np.mean(volume) * 1e-24 
        print(f"Volume: {volume_cm3}")
        sigma_cm = proton_conductivity_cal_f(number_of_proton, D_cm, volume_cm3, int(T))
        log10_sigma_cm = np.log10(sigma_cm)
        D_all.append([int(T), 1000/int(T), D_cm, log10_D_cm, sigma_cm, log10_sigma_cm])
        print(f"MSD at {T}K")
        print(f"MSD: {ave_D*6:.4f} A^2/ps")
        print(f"Diffusion coefficient: {D_cm:.4e} cm^2/s")
        print(f"Proton Conductivity: {sigma_cm:.4e} S/cm")
        print("DONE")
    return D_all

def Arrhenius_plot_f(data_frame, column_name='log10_D', ylbl='Log[D(cm$^{2}$ sec$^{-1}$)]', Ea_show=False, fig_save=False, save_dir='./'):
    df = pd.read_csv(data_frame)
    fig, ax1 = plt.subplots()
    sl, ic, rv, _, _ = stats.linregress(df["1000/T"], df[column_name])
    color = 'k'
    ax1.set_xlabel('1000/T [K$^{-1}$]')
    ax1.set_ylabel(ylbl, color=color)
    ax1.scatter(df["1000/T"], df[column_name], color=color, linewidth=4)
    ax1.plot(df["1000/T"], df["1000/T"]*sl + ic, linestyle='--', color=color, linewidth=4)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create a twin Axes sharing the y-axis
    ax2 = ax1.twiny()
    color = 'k'
    ax2.set_xlabel('Temperature (K)', color=color)
    ax2.plot(df['T(K)'], df[column_name], color=color)
    ax2.tick_params(axis='x', labelcolor=color)
    ax2.invert_xaxis()
    ax2.lines[0].set_visible(False)
    
    fig.tight_layout()
    if Ea_show:
        sl, ic, rv, _, _ = stats.linregress(df["1000/T"], df["log10_D"])
        R = 8.31446261815324  # J/(K·mol)
        E_act = -sl * 1000 * np.log(10) * R * (J / mol)
        text_sentence = f"Ea: {E_act:.2f} eV"
        ax1.text(1.6, df.iloc[-2][column_name], text_sentence, ha='left', va='center', fontsize=font_size, color='red')
    if fig_save:
        plt.savefig(os.path.join(save_dir, f'Arrhenius_plot_{column_name}.png'), facecolor='w', bbox_inches="tight",
                    pad_inches=0.3, transparent=True)
    plt.show()

def Diffusion_Coef_plot(data_frame, column_name='log10_sigma', ylbl='Log[$\sigma$ (S.cm$^{-1}$)]', Ea_show=False, fig_save=False, save_dir='./'):
    df = pd.read_csv(data_frame)
    fig, ax1 = plt.subplots()
    color = 'k'
    ax1.set_xlabel('1000/T [K$^{-1}$]')
    ax1.set_ylabel(ylbl, color=color)
    ax1.plot(df['1000/T'], df[column_name], color=color, linewidth=4)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create a twin Axes sharing the y-axis
    ax2 = ax1.twiny()
    color = 'k'
    ax2.set_xlabel('Temperature (K)', color=color)
    ax2.plot(df['T(K)'], df[column_name], color=color)
    ax2.tick_params(axis='x', labelcolor=color)
    ax2.invert_xaxis()
    ax2.lines[0].set_visible(False)
    
    fig.tight_layout()
    if Ea_show:
        sl, ic, rv, _, _ = stats.linregress(df["1000/T"], df["log10_sigma"])
        R = 8.31446261815324  # J/(K·mol)
        E_act = -sl * 1000 * np.log(10) * R * (J / mol)
        text_sentence = f"Ea: {E_act:.2f} eV"
        ax1.text(1.6, df.iloc[-2][column_name], text_sentence, ha='left', va='center', fontsize=font_size, color='red')
    if fig_save:
        plt.savefig(os.path.join(save_dir, f'diffusion_coef.png'), facecolor='w', bbox_inches="tight",
                    pad_inches=0.3, transparent=True)
    plt.show()

def analyze_MD_traj_log_shifted_window(input_traj, temp, structure_name, atom_diffusion_lbl='H', shift_t=500, window_size=1000, step=1, fig_save=False):
    temp = str(temp)
    traj_file = os.path.join(input_traj, f"MD_{temp}.traj")
    if not os.path.isfile(traj_file):
        print(f'{traj_file} does not exist!')
        return 0
    traj_list = read(traj_file, index=":")
    atom_index = [i for i, x in enumerate(traj_list[0].get_chemical_symbols()) if x == atom_diffusion_lbl]
    print(f"Index of {atom_diffusion_lbl} atom: {atom_index}")
    positions_all = np.array([traj_list[i].get_positions() for i in range(len(traj_list))])
    
    # shape is (n_traj, n_atoms, 3 (xyz))
    print("positions_all.shape: ", positions_all.shape)
    # position of H atom 
    positions = positions_all[:, atom_index]
    positions_x = positions[:, :, 0]
    positions_y = positions[:, :, 1]
    positions_z = positions[:, :, 2]
    
    print("positions.shape    : ", positions.shape)
    print("positions_x.shape  : ", positions_x.shape)
    logging.info(f"positions.shape    : {positions.shape}")
    logging.info(f"positions_x.shape  : {positions_x.shape}")
    # msd for each x,y,z axis
    msd_x = np.mean((positions_x - positions_x[0])**2, axis=1)
    msd_y = np.mean((positions_y - positions_y[0])**2, axis=1)
    msd_z = np.mean((positions_z - positions_z[0])**2, axis=1)
    
    # total msd. sum along xyz axis & mean along H atoms axis.
    msd = np.mean(np.sum((positions - positions[0])**2, axis=2), axis=1)
    fig, ax = plt.subplots()
    ax.plot(range(0, len(msd_x), step), msd_x[::step], label="x")
    ax.plot(range(0, len(msd_y), step), msd_y[::step], label="y")
    ax.plot(range(0, len(msd_z), step), msd_z[::step], label="z")
    ax.grid(True)
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("MSD ($\AA^2$)")
    ax.set_title(f"{structure_name}, xyz MSD at {temp}K")
    ax.legend()
    if fig_save:
        plt.savefig(os.path.join(input_traj, f"MSD_at_{temp}K_xyz.png"), facecolor='w', bbox_inches="tight",
                    pad_inches=0.3, transparent=True)
    plt.show()
    
    fig, ax = plt.subplots()
    ax.plot(range(0, len(msd), step), msd[::step])
    ax.grid()
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("MSD ($\AA^2$)")
    ax.set_title(f"{structure_name}, MSD at {temp}K")
    if fig_save:
        plt.savefig(os.path.join(input_traj, f"MSD_at_{temp}.png"), facecolor='w', bbox_inches="tight",
                    pad_inches=0.3, transparent=True)
    plt.show()
    
    D_list = []
    slope_list = []
    intercept_list = []
    fig, ax = plt.subplots()
    for i in range(0, int(len(msd)/shift_t)):
        msd_t = np.mean(np.sum((positions[i*shift_t:i*shift_t + window_size] - positions[i*shift_t])**2, axis=2), axis=1)
        if len(msd_t) != window_size:
            continue
        slope, intercept, r_value, _, _ = stats.linregress(range(0, window_size, step), msd_t[::step])
        D = slope / 6
        D_list.append(D)
        slope_list.append(slope)
        intercept_list.append(intercept)
        msd_list = msd_t[::step]
        ax.plot(range(0, window_size, step), msd_list, label=f"MSD_{i},  {D:.2f} $\AA^2$/ps")
    plt.grid(True)
    plt.xlabel("time (ps)")
    plt.ylabel("MSD ($\AA^2$)")
    plt.title(f"{structure_name} MSD at {temp}K")
    ax.legend(loc='upper left')
    if fig_save:
        plt.savefig(os.path.join(input_traj, f"MSD_at_{temp}K_shifted_{shift_t}ps_windows_{window_size}.png"), facecolor='w', bbox_inches="tight",
                    pad_inches=0.3, transparent=True)    
    plt.show()
    
    ave_D = np.mean(D_list)
    D_cm = ave_D * 1e-16 / 1e-12
    fig, ax = plt.subplots()
    ax.plot(range(len(msd_list)), np.mean(msd_list, axis=0), label=f"MSD,  {ave_D*6:.2f} $\AA^2$/ps ~ {D_cm:.2e} cm^2/s", linewidth=4)
    ax.plot(range(len(msd_list)), range(len(msd_list)) * np.mean(slope_list) + np.mean(intercept_list), label="fitted line", linewidth=4)
    print(f"Average of MSD: {ave_D*6:.2f} $\AA^2$/ps")
    logging.info(f"Average of Diffusion coefficient: {D_cm:.2e} cm^2/s")
    ax.legend(loc='upper left')
    plt.xlabel("time (ps)")
    plt.ylabel("MSD ($\AA^2$)")
    plt.title(f"{structure_name} MSD at {temp}K")
    text_sentence = f"Diffusion coefficient {D_cm:.3e} cm^2/s"
    ax.text(50, 2, text_sentence, ha='left', va='center', fontsize=20, color='blue')
    if fig_save:
        plt.savefig(os.path.join(input_traj, f"MSD_at_{temp}K_fitted_line_shifted_{shift_t}ps_windows_{window_size}.png"), facecolor='w', bbox_inches="tight",
                    pad_inches=0.3, transparent=True)
    plt.show()
    print(f"MSD {ave_D*6:.3f} $\AA^2$/ps")
    print(f"Diffusion coefficient {D_cm:.3e} cm^2/s")
    logging.info(f"MSD {ave_D*6:.3f} $\AA^2$/ps")
    logging.info(f"Diffusion coefficient {D_cm:.3e} cm^2/s")
    
    # Final MSD Plot with Fitted Line
    t = np.arange(0, len(msd), step)
    slope, intercept, r_value, _, _ = stats.linregress(range(0, len(msd), step), msd[::step])
    D = slope / 6
    D_cm = D * 1e-16 / 1e-12
    fig, ax = plt.subplots()
    ax.plot(t, msd[::step], label="MSD")
    ax.plot(t, t * slope + intercept, label="fitted line")
    ax.grid(True)
    ax.legend(loc='upper left')
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("MSD ($\AA^2$)")
    ax.set_title(f"{structure_name} MSD at {temp}K")
    text_sentence = f"Diffusion Coefficient {D_cm:.3e} cm^2/s"
    ax.text(int(max(t)/2 - 5000), 100, text_sentence, ha='left', va='center', fontsize=20, color='blue')
    if fig_save:
        plt.savefig(os.path.join(input_traj, f"MSD_at_{temp}K_fitted_line.png"), facecolor='w', bbox_inches="tight",
                    pad_inches=0.3, transparent=True)
    plt.show()

def main():
    paths = get_project_paths()

    # Provide the path to the fine-tuned model
    model_path = os.path.join(paths['output_dir'], 'M3GNet_finetuning', 'checkpoints', 'model.pt')

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
    atoms = read(structure_files)

    # Run MD simulation
    results = analyzer.run_md(atoms)

    # Analyze diffusion
    diffusion_coefficient = analyzer.analyze_diffusion(results['positions'])
    print(f"Diffusion Coefficient: {diffusion_coefficient:.6e} cm^2/s")

if __name__ == "__main__":
    # Ignore warnings for cleaner output
    warnings.simplefilter("ignore")
    main()
