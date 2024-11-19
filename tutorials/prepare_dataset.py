import os
import json
import zipfile
import pandas as pd
from pymatgen.core import Structure
from tqdm import tqdm
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_pes
from dgl.data.utils import split_dataset
from functools import partial

def load_dataset(json_path="mp.2018.6.1.json"):
    """Function to load the dataset."""
    if not os.path.exists(json_path):
        from matgl.utils.io import RemoteFile
        url = "https://figshare.com/ndownloader/files/15087992"
        f = RemoteFile(url)
        with zipfile.ZipFile(f.local_path) as zf:
            zf.extractall(".")
    data = pd.read_json(json_path)
    structures = []
    mp_ids = []
    for mid, structure_str in tqdm(zip(data["material_id"], data["structure"]), total=len(data)):
        struct = Structure.from_str(structure_str, fmt="cif")
        structures.append(struct)
        mp_ids.append(mid)
    eform_per_atom = data["formation_energy_per_atom"].tolist()
    return structures, mp_ids, eform_per_atom

def prepare_dataset():
    """Prepare the dataset."""
    structures, mp_ids, eform_per_atom = load_dataset()
    
    # For demonstration, select the first 100 structures
    structures = structures[:100]
    eform_per_atom = eform_per_atom[:100]
    
    # Identify unique elements
    element_types = get_element_list(structures)
    
    # Initialize the graph converter
    converter = Structure2Graph(element_types=element_types, cutoff=5.0)
    
    # Create the dataset
    labels = {
        "energies": eform_per_atom,
        # If forces and stresses are available, add them here
        # "forces": forces,
        # "stresses": stresses,
    }
    
    dataset = MGLDataset(
        threebody_cutoff=4.0,
        structures=structures,
        converter=converter,
        labels=labels,
        include_line_graph=True
    )
    
    # Split the dataset
    train_data, val_data, test_data = split_dataset(
        dataset,
        frac_list=[0.8, 0.1, 0.1],
        shuffle=True,
        random_state=42,
    )
    
    # Define the collate function
    my_collate_fn = partial(collate_fn_pes, include_line_graph=True)
    
    # Create DataLoaders
    train_loader, val_loader, test_loader = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        collate_fn=my_collate_fn,
        batch_size=2,        # Adjust based on hardware
        num_workers=0,       # Set >0 for faster data loading
    )
    
    # Save DataLoaders for later use
    import torch
    torch.save({
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'element_types': element_types
    }, 'data_loaders.pth')
    
    print("Dataset preparation complete and saved as 'data_loaders.pth'.")

if __name__ == "__main__":
    prepare_dataset()
