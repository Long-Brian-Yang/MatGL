import lmdb
import os
import pickle
import subprocess
import torch
# from ase.calculators.emt import EMT
from ase.io import read
# from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
# from ase.md.verlet import VelocityVerlet
from fairchem.core.datasets import LmdbDataset
from fairchem.core.preprocessing import AtomsToGraphs
from tqdm import tqdm

# --- read from VASP OUTCAR
outcar_name = "OUTCAR"
raw_data = read(outcar_name, index=":")
print(f"Number of frames: {len(raw_data)}")

traj_name = "vasp_test"

a2g = AtomsToGraphs(
    max_neigh=100,  # 50
    radius=6,       # do not increase very much
    r_energy=True,  # False for test data
    r_forces=True,  # False for test data
    r_stress=True,
    r_distances=True,
    r_pbc=True,
)

datadir = ""
dataset_name = datadir + traj_name + ".lmdb"

db = lmdb.open(
    dataset_name,
    map_size=1099511627776*2,
    subdir=False,
    meminit=False,
    map_async=True,
)
tags = raw_data[0].get_tags()
data_objects = a2g.convert_all(raw_data, disable_tqdm=True)

for fid, data in tqdm(enumerate(data_objects), total=len(data_objects)):
    data.sid = torch.LongTensor([0])    # assign sid
    data.fid = torch.LongTensor([fid])  # assign fid
    data.tags = torch.LongTensor(tags)  # assign tags, if available

    # no neighbor edge case check
    if data.edge_index.shape[1] == 0:
        print("no neighbors")
        continue

    txn = db.begin(write=True)
    txn.put(f"{fid}".encode("ascii"), pickle.dumps(data, protocol=-1))
    txn.commit()

txn = db.begin(write=True)
txn.put("length".encode("ascii"), pickle.dumps(len(data_objects), protocol=-1))
txn.commit()

db.sync()
db.close()

dataset = LmdbDataset({"src": dataset_name})

subprocess.run("rm *.lmdb-lock", shell=True)  # delete lock file, which is unnecessary

# make directory
train_dir = "../data/s2ef/mytrain"
val_dir   = "../data/s2ef/myval"

for idir in [train_dir, val_dir]:
    if not os.path.exists(idir):
        os.makedirs(idir)

subprocess.run(f"cp {dataset_name} {train_dir}", shell=True)
subprocess.run(f"cp {dataset_name} {val_dir}", shell=True)
