# run_md.py

from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase.md import MDLogger
from ase.io.trajectory import Trajectory
from matgl.ase.calculator import M3GNetCalculator
import matgl
import torch

def run_md(traj_input="BaZrO3_with_H.traj", traj_output="proton_md.traj", log_file="proton_md.log", model_path="./finetuned_m3gnet_model/", dt_fs=1.0, num_steps=10000, thermostat=False, temperature=300, tau=100):
    """Run molecular dynamics simulation."""
    # Load the fine-tuned model
    trained_model = matgl.load_model(path=model_path)
    
    # Set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize the M3GNet calculator
    calc = M3GNetCalculator(model=trained_model, device=device)
    
    # Read the structure with the proton
    atoms = read(traj_input)
    atoms.set_calculator(calc)
    
    # Initialize MD simulation
    dyn = VelocityVerlet(atoms, dt=dt_fs * 1e-15)  # Convert fs to seconds
    
    # Set up the logger
    logger = MDLogger(dyn, atoms, log_file, header=True, stress=False)
    dyn.attach(logger, interval=10)         # Log every 10 steps
    
    # Set up trajectory writer
    traj_writer = Trajectory(traj_output, 'w')
    dyn.attach(traj_writer, interval=10)    # Write trajectory every 10 steps
    
    # If thermostat is enabled (NVT ensemble)
    if thermostat:
        from ase.md.nose_hoover import NoseHoover
        thermo = NoseHoover(atoms, temperature=temperature, tau=tau)
        dyn.attach(thermo, interval=100)    # Update thermostat every 100 steps
    
    # Run MD simulation
    dyn.run(num_steps)
    
    print(f"Molecular dynamics simulation complete. Trajectory saved to '{traj_output}', log saved to '{log_file}'.")

if __name__ == "__main__":
    run_md()
