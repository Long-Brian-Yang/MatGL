import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory

def calculate_msd(traj_file="proton_md.traj"):
    """Calculate Mean Square Displacement (MSD) and diffusion coefficient (D)."""
    # Load the trajectory
    traj = Trajectory(traj_file)
    
    # Initialize position list
    positions = []
    
    for frame in traj:
        positions.append(frame.positions.copy())
    
    positions = np.array(positions)  # Shape: (num_steps, num_atoms, 3)
    
    # Identify the hydrogen atom index
    h_index = None
    atoms = traj[0]
    for i, atom in enumerate(atoms):
        if atom.symbol == 'H':
            h_index = i
            break
    
    if h_index is None:
        raise ValueError("No hydrogen atom found in the trajectory.")
    
    # Extract hydrogen positions over time
    h_positions = positions[:, h_index, :]  # Shape: (num_steps, 3)
    
    # Calculate MSD
    initial_position = h_positions[0]
    displacements = h_positions - initial_position
    msd = np.mean(displacements**2, axis=1)
    
    # Define time steps (in picoseconds)
    dt_fs = 1.0  # Time step in fs (as set in run_md.py)
    time_steps = np.arange(len(msd)) * dt_fs * 1e-3  # Convert fs to ps
    
    # Select the linear region for fitting (e.g., last 50% of the simulation)
    start_fit = len(msd) // 2
    X = sm.add_constant(time_steps[start_fit:])
    Y = msd[start_fit:]
    
    # Perform linear regression
    model = sm.OLS(Y, X)
    results = model.fit()
    slope = results.params[1]
    D = slope / 6  # Diffusion coefficient in Å²/ps
    
    print(f"Diffusion Coefficient D = {D:.3f} Å²/ps")
    
    # Plot MSD and the linear fit
    plt.figure(figsize=(8, 6))
    plt.plot(time_steps, msd, label='MSD')
    plt.plot(time_steps[start_fit:], results.fittedvalues, label='Linear Fit', linestyle='--')
    plt.xlabel('Time (ps)')
    plt.ylabel('MSD (Å²)')
    plt.title('Mean Square Displacement of Proton')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    calculate_msd()
