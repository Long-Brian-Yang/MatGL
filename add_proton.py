# add_proton.py

from ase.io import read, write
from ase.build import add_adsorbate
from matgl.ase.calculator import M3GNetCalculator
import matgl
import torch

def add_proton(traj_input="optimized_BaZrO3.traj", traj_output="BaZrO3_with_H.traj", model_path="./finetuned_m3gnet_model/", position='ontop', height=1.0):
    """Add a proton to the optimized structure."""
    # Load the fine-tuned model
    trained_model = matgl.load_model(path=model_path)
    
    # Set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize the M3GNet calculator
    calc = M3GNetCalculator(model=trained_model, device=device)
    
    # Read the optimized structure
    atoms = read(traj_input)
    
    # Add a hydrogen atom (proton)
    add_adsorbate(atoms, 'H', height=height, position=position)
    
    # Reassign the calculator
    atoms.set_calculator(calc)
    
    # Save the structure with the proton
    write(traj_output, atoms)
    
    print(f"Proton added. New trajectory saved to '{traj_output}'.")

if __name__ == "__main__":
    add_proton()
