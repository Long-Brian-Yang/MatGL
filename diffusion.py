from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import MDAnalysis as mda
from MDAnalysis.analysis import msd
import json
from concurrent.futures import ProcessPoolExecutor, as_completed


class DiffusionAnalyzer:
    """Analyzer for diffusion coefficients and proton conductivity"""

    def __init__(self, log_file: Path, md_config: Optional[Dict] = None):
        # Physical constants
        self.e = 1.60217662e-19  # Elementary charge (C)
        self.k_B = 1.38064852e-23  # Boltzmann constant (J/K)

        # Setup logging
        handler = RotatingFileHandler(
            log_file, maxBytes=10**6, backupCount=5
        )
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        self.logger = logging.getLogger('DiffusionAnalyzer')
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

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

    def load_trajectory(self, traj_file: str) -> mda.Universe:
        """Load trajectory file"""
        try:
            u = mda.Universe(traj_file)
            self.logger.info(f"Successfully loaded trajectory file: {traj_file}")
            return u
        except Exception as e:
            self.logger.error(f"Error loading trajectory file {traj_file}: {str(e)}")
            raise

    def select_atoms(self, universe: mda.Universe, atom_type: str) -> mda.AtomGroup:
        """Select specified type of atoms"""
        selection = universe.select_atoms(f'type {atom_type}')
        if not selection:
            raise ValueError(f"Atom type '{atom_type}' not found in trajectory file")
        self.logger.info(f"Selected atom type: {atom_type}, Count: {len(selection)}")
        return selection

    def analyze_trajectory(
            self,
            traj_file: str,
            temperature: float,
            atom_type: str = 'H',
            time_step_fs: float = 1.0
        ) -> Dict:
        """
        Analyze trajectory file to calculate MSD and diffusion coefficient

        Args:
            traj_file: Path to trajectory file
            temperature: Temperature (K)
            atom_type: Atom type (default 'H')
            time_step_fs: Time step in femtoseconds (default 1.0)

        Returns:
            Analysis result dictionary
        """
        try:
            # Load trajectory
            u = self.load_trajectory(traj_file)

            # Select specified type of atoms
            selection = self.select_atoms(u, atom_type)

            num_protons = len(selection)
            self.logger.info(f"Analyzing trajectory file: {traj_file}")
            self.logger.info(f"Selected atom type: {atom_type}, Count: {num_protons}")

            # Calculate volume (assuming NVT ensemble, constant volume)
            volumes = []
            for ts in u.trajectory:
                vol = np.linalg.det(ts.dimensions[:3])  # Exclude angles
                volumes.append(vol)
            avg_volume_cm3 = np.mean(volumes) * 1e-24  # Å³ -> cm³
            self.logger.info(f"Average volume: {avg_volume_cm3:.4e} cm³")

            # Calculate MSD and diffusion coefficient
            msd_analysis = msd.MSD(selection, msdtype='xyz')
            msd_analysis.run()
            D = msd_analysis.results.diffusion_coefficient * 1e-16  # Convert to cm²/s
            self.logger.info(f"Diffusion coefficient: {D:.4e} cm²/s")

            # Calculate conductivity
            sigma = self.calculate_conductivity(
                n_protons=num_protons,
                diffusion_coef=D,
                volume=avg_volume_cm3,
                temperature=temperature
            )
            self.logger.info(f"Proton conductivity: {sigma:.4e} S/cm")

            # Handle logarithmic calculations to avoid negative infinity
            log10_D = np.log10(D) if D > 0 else -np.inf
            log10_sigma = np.log10(sigma) if sigma > 0 else -np.inf

            # Get MSD and time data
            times = msd_analysis.results.times  # Time array (ps)
            msd_values = msd_analysis.results.msd  # MSD array (Å²)

            return {
                'T(K)': temperature,
                '1000/T': 1000 / temperature,
                'D(cm²/s)': D,
                'log10_D': log10_D,
                'sigma(S/cm)': sigma,
                'log10_sigma': log10_sigma,
                'volume_cm³': avg_volume_cm3,
                'msd_times_ps': times,
                'msd_values_A2': msd_values
            }

        except Exception as e:
            self.logger.error(f"Error analyzing trajectory file {traj_file}: {str(e)}")
            return {}

    def calculate_conductivity(
            self,
            n_protons: int,
            diffusion_coef: float,
            volume: float,
            temperature: float
        ) -> float:
        """
        Calculate proton conductivity

        Args:
            n_protons: Number of protons
            diffusion_coef: Diffusion coefficient (cm²/s)
            volume: Volume (cm³)
            temperature: Temperature (K)

        Returns:
            Conductivity (S/cm)
        """
        try:
            conductivity = (n_protons * self.e * diffusion_coef) / (self.k_B * temperature * volume)
            self.logger.info(f"Calculated conductivity: {conductivity:.4e} S/cm")
            return conductivity
        except Exception as e:
            self.logger.error(f"Error calculating conductivity: {str(e)}")
            return 0.0

    def extract_temperature(self, traj_name: str) -> Optional[float]:
        """
        Extract temperature information from trajectory name

        Args:
            traj_name: Name of the trajectory file

        Returns:
            Temperature value (K) or None
        """
        try:
            # Assume trajectory name contains temperature information, e.g., "MD_300K.traj"
            import re
            match = re.search(r'(\d+)K', traj_name)
            if match:
                return float(match.group(1))
            else:
                self.logger.warning(f"Unable to extract temperature information from trajectory name {traj_name}.")
                return None
        except Exception as e:
            self.logger.error(f"Error extracting temperature from trajectory name {traj_name}: {str(e)}")
            return None


class PlotManager:
    """Plotting Manager"""

    def __init__(self, font_size: int = 26):
        self.font_size = font_size
        self._setup_style()
        self.logger = logging.getLogger('PlotManager')  # Add logger

    def _setup_style(self):
        """Setup plotting style"""
        plt.rcParams.update({
            'font.size': self.font_size,
            'font.family': 'DejaVu Sans',
            'legend.fontsize': self.font_size,
            'xtick.labelsize': self.font_size,
            'ytick.labelsize': self.font_size,
            'axes.labelsize': self.font_size,
            'axes.titlesize': self.font_size,
            'figure.figsize': [14, 14]
        })

    def plot_arrhenius(
            self,
            data: Dict[str, List[float]],
            save_dir: str,
            column_name: str = 'log10_D',
            y_label: str = 'log[D(cm$^2$ s$^{-1}$)]',
            show_ea: bool = True
        ):
        """
        Plot Arrhenius graph

        Args:
            data: Analysis result dictionary, format {'T(K)': [...], '1000/T': [...], 'log10_D': [...], ...}
            save_dir: Save directory
            column_name: Column name to plot
            y_label: Y-axis label
            show_ea: Whether to display activation energy
        """
        try:
            temperatures = data['T(K)']
            inv_temperatures = data['1000/T']
            y_values = data[column_name]

            # Linear fit
            slope, intercept, r_value, p_value, std_err = stats.linregress(inv_temperatures, y_values)

            fig, ax1 = plt.subplots()

            ax1.plot(inv_temperatures, y_values, 'o', label=column_name)
            ax1.plot(inv_temperatures, intercept + slope * np.array(inv_temperatures), 'r', label='Fit Line')

            ax1.set_xlabel('1000/T [K$^{-1}$]')
            ax1.set_ylabel(y_label)
            ax1.set_title('Arrhenius Plot')
            ax1.legend()

            if show_ea:
                # Calculate activation energy E_a = slope * R
                R = 8.314  # J/(mol·K)
                E_a = slope * R  # Unit J/mol
                E_a_kJ_mol = E_a / 1000  # Convert to kJ/mol
                plt.text(0.05, 0.95, f'Eₐ = {E_a_kJ_mol:.2f} kJ/mol', transform=ax1.transAxes,
                         fontsize=self.font_size, verticalalignment='top')

            plt.tight_layout()
            plt.savefig(Path(save_dir) / 'arrhenius_plot.png')
            plt.close()
            logging.getLogger('PlotManager').info("Arrhenius plot saved to arrhenius_plot.png")

        except Exception as e:
            logging.getLogger('PlotManager').error(f"Error plotting Arrhenius graph: {str(e)}")

    def plot_msd_time(
            self,
            msd_data: Dict[str, Dict],
            save_dir: str
        ):
        """
        Plot MSD vs Time graph

        Args:
            msd_data: MSD analysis result dictionary, format {traj_name: {'msd_times_ps': [...], 'msd_values_A2': [...], ...}, ...}
            save_dir: Save directory
        """
        try:
            for traj_name, data in msd_data.items():
                times = data.get('msd_times_ps', [])
                msd = data.get('msd_values_A2', [])

                if not times or not msd:
                    logging.getLogger('PlotManager').warning(f"{traj_name} is missing MSD or time data, skipping plot.")
                    continue

                plt.figure()
                plt.plot(times, msd, 'o-', label='MSD')
                plt.xlabel('Time (ps)')
                plt.ylabel('Mean Squared Displacement (Å$^2$)')
                plt.title(f'MSD vs Time for {traj_name}')
                plt.legend()
                plt.tight_layout()
                plot_path = Path(save_dir) / f'msd_time_{traj_name}.png'
                plt.savefig(plot_path)
                plt.close()
                logging.getLogger('PlotManager').info(f"MSD vs Time plot saved to {plot_path}")

        except Exception as e:
            logging.getLogger('PlotManager').error(f"Error plotting MSD vs Time graph: {str(e)}")

    def plot_diffusion_vs_temperature(
            self,
            diffusion_data: Dict[str, float],
            save_dir: str
        ):
        """
        Plot diffusion coefficient vs temperature graph

        Args:
            diffusion_data: Diffusion coefficient dictionary, format {traj_name: D, ...}
            save_dir: Save directory
        """
        try:
            temperatures = []
            diffusion_coeffs = []

            for traj_name, D in diffusion_data.items():
                temp = self.extract_temperature(traj_name)
                if temp is not None:
                    temperatures.append(temp)
                    diffusion_coeffs.append(D)

            plt.figure()
            plt.plot(temperatures, diffusion_coeffs, 's-', color='green', label='D(cm²/s)')
            plt.xlabel('Temperature (K)')
            plt.ylabel('Diffusion Coefficient (cm²/s)')
            plt.title('Diffusion Coefficient vs Temperature')
            plt.legend()
            plt.tight_layout()
            plot_path = Path(save_dir) / 'diffusion_vs_temperature.png'
            plt.savefig(plot_path)
            plt.close()
            logging.getLogger('PlotManager').info(f"Diffusion Coefficient vs Temperature plot saved to {plot_path}")

        except Exception as e:
            logging.getLogger('PlotManager').error(f"Error plotting Diffusion Coefficient vs Temperature graph: {str(e)}")

    def plot_conductivity_vs_temperature(
            self,
            conductivity_data: Dict[str, float],
            save_dir: str
        ):
        """
        Plot conductivity vs temperature graph

        Args:
            conductivity_data: Conductivity dictionary, format {traj_name: sigma, ...}
            save_dir: Save directory
        """
        try:
            temperatures = []
            conductivities = []

            for traj_name, sigma in conductivity_data.items():
                temp = self.extract_temperature(traj_name)
                if temp is not None:
                    temperatures.append(temp)
                    conductivities.append(sigma)

            plt.figure()
            plt.plot(temperatures, conductivities, 'D--', color='blue', label='σ(S/cm)')
            plt.xlabel('Temperature (K)')
            plt.ylabel('Conductivity (S/cm)')
            plt.title('Conductivity vs Temperature')
            plt.legend()
            plt.tight_layout()
            plot_path = Path(save_dir) / 'conductivity_vs_temperature.png'
            plt.savefig(plot_path)
            plt.close()
            logging.getLogger('PlotManager').info(f"Conductivity vs Temperature plot saved to {plot_path}")

        except Exception as e:
            logging.getLogger('PlotManager').error(f"Error plotting Conductivity vs Temperature graph: {str(e)}")

    def plot_multi_trajectory_comparison(
            self,
            msd_data: Dict[str, Dict],
            save_dir: str
        ):
        """
        Plot MSD comparison for multiple trajectories

        Args:
            msd_data: MSD analysis result dictionary, format {traj_name: {'msd_times_ps': [...], 'msd_values_A2': [...], ...}, ...}
            save_dir: Save directory
        """
        try:
            plt.figure()
            for traj_name, data in msd_data.items():
                times = data.get('msd_times_ps', [])
                msd = data.get('msd_values_A2', [])

                if not times or not msd:
                    logging.getLogger('PlotManager').warning(f"{traj_name} is missing MSD or time data, skipping plot.")
                    continue

                plt.plot(times, msd, label=traj_name)

            plt.xlabel('Time (ps)')
            plt.ylabel('Mean Squared Displacement (Å$^2$)')
            plt.title('MSD Comparison for Multiple Trajectories')
            plt.legend()
            plt.tight_layout()
            plot_path = Path(save_dir) / 'multi_trajectory_msd_comparison.png'
            plt.savefig(plot_path)
            plt.close()
            logging.getLogger('PlotManager').info(f"MSD comparison for multiple trajectories plot saved to {plot_path}")

        except Exception as e:
            logging.getLogger('PlotManager').error(f"Error plotting MSD comparison for multiple trajectories: {str(e)}")

    def extract_temperature(self, traj_name: str) -> Optional[float]:
        """
        Extract temperature information from trajectory name

        Args:
            traj_name: Name of the trajectory file

        Returns:
            Temperature value (K) or None
        """
        try:
            # Assume trajectory name contains temperature information, e.g., "MD_300K.traj"
            import re
            match = re.search(r'(\d+)K', traj_name)
            if match:
                return float(match.group(1))
            else:
                self.logger.warning(f"Unable to extract temperature information from trajectory name {traj_name}.")
                return None
        except Exception as e:
            self.logger.error(f"Error extracting temperature from trajectory name {traj_name}: {str(e)}")
            return None


def load_config(config_path: str) -> Dict:
    """
    Load configuration file

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found. Please ensure the file exists at the correct path.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Configuration file {config_path} has invalid format. Please check the JSON syntax.")
        exit(1)


def process_traj(traj_file: str, config: Dict) -> Dict:
    """
    Analyze a single trajectory file

    Args:
        traj_file: Path to trajectory file
        config: Configuration dictionary

    Returns:
        Analysis result dictionary
    """
    analyzer = DiffusionAnalyzer(log_file=Path(config['log_file']), md_config=config.get('md_config'))
    
    # Extract temperature
    temp = analyzer.extract_temperature(Path(traj_file).name)
    if temp is None:
        analyzer.logger.error(f"Unable to extract temperature from filename: {traj_file}")
        return {Path(traj_file).name: {}}
    
    result = analyzer.analyze_trajectory(
        traj_file=traj_file,
        temperature=temp,
        atom_type=config.get('atom_type', 'H'),
        time_step_fs=config.get('time_step_fs', 1.0)
    )
    return {Path(traj_file).name: result}


def visualize_results(diffusion_results: Dict, plot_output_dir: str):
    """
    Visualize analysis results

    Args:
        diffusion_results: Analysis result dictionary
        plot_output_dir: Directory to save plots
    """
    try:
        plot_manager = PlotManager(font_size=26)

        # Plot Arrhenius graph
        arrhenius_data = {
            'T(K)': [],
            '1000/T': [],
            'log10_D': []
        }
        for traj_name, data in diffusion_results.items():
            arrhenius_data['T(K)'].append(data.get('T(K)', 0))
            arrhenius_data['1000/T'].append(data.get('1000/T', 0))
            arrhenius_data['log10_D'].append(data.get('log10_D', -np.inf))

        plot_manager.plot_arrhenius(
            data=arrhenius_data,
            save_dir=plot_output_dir,
            column_name='log10_D',
            y_label='log[D(cm$^2$ s$^{-1}$)]',
            show_ea=True
        )

        # Plot MSD vs Time graph
        plot_manager.plot_msd_time(
            msd_data=diffusion_results,
            save_dir=plot_output_dir
        )

        # Collect diffusion coefficient and conductivity data
        diffusion_data = {}
        conductivity_data = {}
        for traj_name, data in diffusion_results.items():
            diffusion_data[traj_name] = data.get('D(cm²/s)', 0)
            conductivity_data[traj_name] = data.get('sigma(S/cm)', 0)

        # Plot diffusion coefficient vs temperature graph
        plot_manager.plot_diffusion_vs_temperature(
            diffusion_data=diffusion_data,
            save_dir=plot_output_dir
        )

        # Plot conductivity vs temperature graph
        plot_manager.plot_conductivity_vs_temperature(
            conductivity_data=conductivity_data,
            save_dir=plot_output_dir
        )

        # Plot MSD comparison for multiple trajectories
        plot_manager.plot_multi_trajectory_comparison(
            msd_data=diffusion_results,
            save_dir=plot_output_dir
        )

    except Exception as e:
        logging.getLogger('PlotManager').error(f"Error visualizing results: {str(e)}")


def main():
    # Load configuration
    config = load_config('config_diffusion.json')

    # Get list of trajectory files
    trajectories_dir = Path(config['trajectories_dir'])
    traj_files = list(trajectories_dir.glob(config.get('traj_pattern', '*.traj')))  # Adjust according to actual trajectory format

    diffusion_results = {}

    # Parallel processing of trajectory files
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_traj,
                traj_file=str(traj),
                config=config
            ): traj for traj in traj_files
        }

        for future in as_completed(futures):
            result = future.result()
            diffusion_results.update(result)

    # Save results
    diffusion_results_path = Path(config.get('diffusion_results', 'diffusion_results.json'))
    with open(diffusion_results_path, 'w') as f:
        json.dump(diffusion_results, f, indent=4)
    print(f"Diffusion analysis results saved to {diffusion_results_path}")

    # Create plot output directory
    plot_output_dir = Path(config.get('plot_output_dir', 'plots'))
    plot_output_dir.mkdir(exist_ok=True)

    # Visualize results
    visualize_results(diffusion_results, str(plot_output_dir))
    print(f"Visualization plots saved to {plot_output_dir}")


if __name__ == "__main__":
    main()