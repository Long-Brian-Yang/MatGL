from ase.cluster import Icosahedron
from ase.visualize import view
import numpy as np
from ase.io import write


def replace_element(atoms, from_element, to_element, percent_replace=100):
    import random

    from ase.build import sort

    elements = atoms.get_chemical_symbols()
    num_from_elements = elements.count(from_element)
    num_replace = int((percent_replace/100) * num_from_elements)

    indices = [i for i, j in enumerate(elements) if j == from_element]
    random_item = random.sample(indices, num_replace)
    for i in random_item:
        atoms[i].symbol = to_element

    atoms = sort(atoms)
    return atoms


size = "small"  # "small" or "large"
noshells = 3 if size == "small" else 6

atoms = Icosahedron("Pt", noshells=noshells, latticeconstant=3.92)
atoms = replace_element(atoms=atoms, from_element="Pt", to_element="Pd", percent_replace=20)
atoms = replace_element(atoms=atoms, from_element="Pt", to_element="Ni", percent_replace=20)
atoms = replace_element(atoms=atoms, from_element="Pt", to_element="Cu", percent_replace=20)

atoms.cell = [20, 20, 20]
atoms.translate([10, 10, 4.0])
pos  = atoms.get_positions()
pos2 = np.array(sorted(pos, key=lambda x: x[2]))
atoms.set_positions(pos2)

thre = 0.0
delete_list = list(filter(lambda x: x[2] < thre, atoms.get_positions()))
del atoms[0:len(delete_list)]
print(f"delted {len(delete_list)} atoms, remaining {len(atoms)} atoms")

view(atoms)

write("POSCAR_cluster", atoms)
