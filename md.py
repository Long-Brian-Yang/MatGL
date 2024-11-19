import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matgl.models import M3GNet
from pymatgen.core import Structure

def calculate_msd(positions, reference_pos):
    """Calculate Mean Square Displacement"""
    return np.mean(np.sum((positions - reference_pos) ** 2, axis=1))

def run_md_simulation(model, structure, steps=1000, dt=0.001):
    """Run MD simulation and calculate MSD"""
    positions = []
    initial_pos = structure.cart_coords.copy()
    current_pos = initial_pos.copy()
    
    msd_values = []
    time_steps = []
    
    for step in tqdm(range(steps)):
        # Get forces from model
        forces = model.predict_forces(structure)
        
        # Update positions using velocity Verlet
        current_pos += forces * dt**2
        
        # Update structure
        structure.cart_coords = current_pos
        
        # Calculate MSD
        msd = calculate_msd(current_pos, initial_pos)
        
        positions.append(current_pos.copy())
        msd_values.append(msd)
        time_steps.append(step * dt)
    
    return np.array(time_steps), np.array(msd_values)

def plot_msd_comparison(pretrained_results, finetuned_results):
    """Plot MSD comparison between pre-trained and fine-tuned models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pre-trained model results
    ax1.plot(pretrained_results[0], pretrained_results[1], 'b-')
    ax1.set_xlabel('Time (ps)')
    ax1.set_ylabel('MSD (Angstrom^2)')
    ax1.set_title('M3GNet, pre-trained')
    
    # Fine-tuned model results
    ax2.plot(finetuned_results[0], finetuned_results[1], 'b-')
    ax2.set_xlabel('Time (ps)')
    ax2.set_ylabel('MSD (Angstrom^2)')
    ax2.set_title('M3GNet, fine-tuned by VASP')
    
    plt.tight_layout()
    plt.savefig('msd_comparison.png', dpi=300, bbox_inches='tight')

def main():
    # Load models
    pretrained_model = M3GNet.load('path_to_pretrained_model')
    finetuned_model = M3GNet.load('path_to_finetuned_model')
    
    # Load test structure
    structure = Structure.from_file('test_structure.vasp')
    
    # Run simulations
    pre_time, pre_msd = run_md_simulation(pretrained_model, structure)
    fine_time, fine_msd = run_md_simulation(finetuned_model, structure)
    
    # Plot results
    plot_msd_comparison(
        (pre_time, pre_msd),
        (fine_time, fine_msd)
    )

if __name__ == "__main__":
    main()