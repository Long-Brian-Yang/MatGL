### Overview

The framework includes two versions: `pre-train` and `finetune`. The `dataset` and `diffusion` components are shared between the two versions. The current test run only loads data for 100 materials.

### Pre-train Version

1. **Execution Flow**  
   Directly run `pretrained_md` to generate MD simulation results, followed by `diffusion` for diffusion analysis.

2. **Code Details**

   - **Parameter Setup**

     ```python
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
             'temperatures': [600]  # Unit: K
         }
     }
     ```

   - **Key Points**
     - `md_params`: Adjusts the pre-train model and MD simulation settings.
     - `simulation`: Supports changing the structure file, adjusting the number of protons (`n_protons`), and setting simulation temperatures (`temperatures`).
     - The test structure file `Ba8Zr8O24` is located in the `data` folder instead of `data/structures`.

3. **Output Results**
   - MD trajectory files are stored in `logs/md_trajectories`.
   - Diffusion analysis results are saved in `logs/md_analysis`.

---

### Finetune Version

1. **Execution Flow**  
   Run the following steps in order:

   - `finetune`
   - `md`
   - `diffusion`

2. **Code Details**

   - Run `finetune` to generate a fine-tuned model.
   - In the `md` step, load the fine-tuned model and perform MD simulation.
   - Use `diffusion` to analyze diffusion behavior.

3. **Key Notes**
   - The settings for the `md` module are identical to those in the pre-train version.
   - The `diffusion` analysis component is shared between both versions.

---

### Summary and Issues

1. The current code is lengthy and lacks proper formatting, requiring further refinement.
2. Pre-training and relaxation (relax) steps are not included and could be added later.
3. The `label` and `dataset` settings differ from the official model, causing difficulties in some functions, particularly in the finetune component, which needs fix.

---

## Framework Core Functions(dataset, md, diffusion)

## 1. Data Processing (DataProcessor Class)

### Basic Configuration Functions

```python
def get_project_paths():
    """Retrieve project path configurations"""

def __init__(self, config: dict):
    """Initialize the data processor"""
```

### Data Loading and Processing Functions

```python
def read_poscar(self, file_path: str) -> Structure:
    """Read a POSCAR file"""

def load_data(self, bandgap_column: str = 'Bandgap_by_DFT') -> None:
    """Load structure and bandgap data"""

def create_dataset(self, normalize: bool = False) -> MGLDataset:
    """Create a graph dataset"""

def create_dataloaders(self) -> Tuple[MGLDataLoader, MGLDataLoader, MGLDataLoader]:
    """Create data loaders"""
```

## 2. Molecular Dynamics Simulation (MDSystem Class)

### System Initialization and Environment Setup

```python
def __init__(self, config: dict, model_checkpoint: str, ...):
    """Initialize the MD system"""

def setup_environment(self):
    """Set up environment and logging"""

def load_finetuned_model(self, checkpoint_path: str):
    """Load a fine-tuned model"""
```

### Structure Handling Functions

```python
def find_vasp_files(self) -> List[Path]:
    """Locate VASP files"""

def add_protons(self, atoms: Atoms, n_protons: int) -> Atoms:
    """Add protons to the structure"""
```

### Core MD Simulation Functions

```python
def run_md(self, structure_file: Path, temperature: float, traj_file: Optional[Path]) -> str:
    """Run a single MD simulation"""

def run_temperature_range(self, structure_files: List[Path], temperatures: List[float]) -> Dict:
    """Run MD simulations across a temperature range"""
```

## 3. Diffusion Analysis (MDAnalysisSystem Class)

### System Initialization

```python
def __init__(self, config: dict, structure_name: str, target_atom: str, ...):
    """Initialize the analysis system"""

def setup_environment(self):
    """Set up environment and logging"""

def setup_plot_parameters(self):
    """Configure plot parameters"""
```

### Trajectory Analysis Functions

```python
def analyze_single_trajectory(self, temperature: int) -> dict:
    """Analyze a single trajectory"""

def analyze_temperature_range(self, temperatures: list = None) -> pd.DataFrame:
    """Analyze trajectories across a temperature range"""
```

### Physical Property Calculation Functions

```python
def calculate_conductivity(self, D_cm2_s: float, temperature: float, volume_cm3: float, n_carriers: int) -> float:
    """Calculate ionic conductivity"""
```

### Visualization and Result Output Functions

```python
def plot_msd_components(self, time_array, msd_data, temperature, filename):
    """Plot MSD components"""

def plot_average_msd(self, time_array, avg_msd, D_cm2_s, temperature, filename):
    """Plot average MSD"""

def plot_arrhenius(self, df: pd.DataFrame, plot_type: str):
    """Plot Arrhenius graph"""

def export_summary(self, df: pd.DataFrame) -> str:
    """Export analysis summary"""
```

### Helper Functions

```python
def run_analysis(analyzer: MDAnalysisSystem, temperatures: list):
    """Run diffusion analysis"""

def print_results(results_df: pd.DataFrame):
    """Print analysis results"""

def save_activation_energy(results_df: pd.DataFrame, structure_name: str):
    """Save activation energy results"""
```

## 4. Main Workflow Function

```python
def main():
    """Main function to coordinate the entire workflow"""
```

## Function Call Relationships

1. **Data Processing Workflow**:

   - `get_project_paths` → `DataProcessor.__init__` → `load_data` → `create_dataset` → `create_dataloaders`

2. **MD Simulation Workflow**:

   - `MDSystem.__init__` → `setup_environment` → `load_finetuned_model` → `find_vasp_files` → `add_protons` → `run_md`/`run_temperature_range`

3. **Diffusion Analysis Workflow**:
   - `MDAnalysisSystem.__init__` → `setup_environment` → `analyze_single_trajectory`/`analyze_temperature_range` → `calculate_conductivity` → `plot_*` → `export_summary`
