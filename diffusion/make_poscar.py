from ase import Atom
from ase.io import read, write
from ase.build import sort


def replace_element(atoms, from_element, to_element, percent_replace):
    import random

    elements = atoms.get_chemical_symbols()
    num_from_elements = elements.count(from_element)
    num_replace = int((percent_replace/100) * num_from_elements)

    indices = [i for i, j in enumerate(elements) if j == from_element]
    random_item = random.sample(indices, num_replace)
    for i in random_item:
        atoms[i].symbol = to_element

    atoms = sort(atoms)
    return atoms


bulk = read("BaZrO3.cif")
replicate_size = 3
replicate = [replicate_size]*3
bulk = bulk*replicate
bulk = sort(bulk)
cell_length = bulk.cell.cellpar()
xpos = 0.50 * cell_length[0] / replicate_size
bulk.append(Atom("H", position=[xpos, 0, 0]))

bulk = replace_element(bulk, from_element="Zr", to_element="Y", percent_replace=0)

write("POSCAR", bulk)
