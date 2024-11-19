from ase.io import read, write
from ase.visualize import view
from ase.build import stack, add_vacuum, surface, sort
import numpy as np

# small
repeat = [5, 5, 1]
layers = 2

# large
# repeat = [10, 10, 1]
# layers = 4

mol = read("../data/BaZrO3.cif")
surf = surface(mol, indices=[0, 0, 1], layers=layers, vacuum=9.0)

low = surf * repeat

high = read("../POSCAR_cluster")
high.cell = low.cell

stacked = stack(low, high, maxstrain=2.0, distance=2.5)

lowest_z = np.min(stacked.get_positions()[:, 2])
highest_z = np.max(stacked.get_positions()[:, 2])
delta_z = highest_z - lowest_z

vacuum = 12.0

stacked.translate([0, 0, -lowest_z + 1.0])
stacked.set_cell([low.cell[0][0], low.cell[1][1], delta_z + vacuum])
stacked.pbc = True
stacked = sort(stacked)

view(stacked)
write("../POSCAR_stacked", stacked)
