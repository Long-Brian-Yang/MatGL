from ase.io import read
from ase.optimize import BFGS
from matgl.ase.calculator import M3GNetCalculator
import matgl
import subprocess
import torch

def optimize_structure(cif_input="BaZrO3.cif", traj_output="optimized_BaZrO3.traj", model_path="./finetuned_m3gnet_model/"):
    """Optimize the BaZrO3 structure."""
    # Load the fine-tuned model
    trained_model = matgl.load_model(path=model_path)
    
    # Set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize the M3GNet calculator
    calc = M3GNetCalculator(model=trained_model, device=device)
    
    # Load the BaZrO3 structure
    structure = read(cif_input)
    structure.set_calculator(calc)
    
    # Initialize the optimizer
    optimizer = BFGS(structure, trajectory=traj_output)
    
    # Run the optimization
    optimizer.run(fmax=0.05, steps=100)
    
    print(f"Structure optimization complete. Trajectory saved to '{traj_output}'.")
    
    # Optional: Visualize the optimized structure
    subprocess.run(f"ase gui {traj_output}", shell=True)

if __name__ == "__main__":
    optimize_structure()
