import os
import subprocess
import time
# from fairchem.core.common.tutorial_utils import fairchem_main
from fairchem.core.common.tutorial_utils import generate_yml_config
from fairchem.core.common.tutorial_utils import train_test_val_split
# os.environ['OPENBLAS_NUM_THREADS'] = '1'  # if openblas error happens, uncomment this
from fairchem.core.models import model_name_to_local_file

# when clean checkpoint file
subprocess.run("rm -rf ./checkpoints/*", shell=True)

# pretrained_model = "DimeNet++-S2EF-ODAC"  # bad
pretrained_model = "PaiNN-S2EF-OC20-All"
# pretrained_model = "GemNet-OC-S2EFS-OC20+OC22"  # takes time
# pretrained_model = "GemNet-OC-S2EFS-OC22"

checkpoint_path = model_name_to_local_file(model_name=pretrained_model, local_cache="../pretrained_checkpoints")

use_asedb = False  # when using ASE database

update = {"gpus": 0,
          "trainer": "ocp",

          "optim.eval_every": 1,
          "optim.max_epochs": 1,  # 10,
          "optim.num_workers": 0,
          "optim.batch_size": 40,  # Number of samples in one batch. Smaller is accurate but takes time. Max is data/5.

          "logger": "tensorboard",

          "dataset.train.a2g_args.r_energy": True,
          "dataset.train.a2g_args.r_forces": True,

          "dataset.val.a2g_args.r_energy": True,
          "dataset.val.a2g_args.r_forces": True,
          }

if use_asedb:
    subprocess.run("rm -rf train.db test.db val.db *.db.lock", shell=True)
    train, test, val = train_test_val_split("bulk.db")  # when using ASE database
    update.update({"task.dataset": "ase_db"})
    update.update({"dataset.train.src": "train.db"})
    update.update({"dataset.test.src": "test.db"})
    update.update({"dataset.test.a2g_args.r_energy": False})
    update.update({"dataset.test.a2g_args.r_forces": False})
    update.update({"dataset.val.src": "val.db"})
else:
    update.update({"task.dataset": "lmdb"})
    update.update({"dataset.train.src": "../data/s2ef/mytrain"})
    update.update({"dataset.val.src": "../data/s2ef/myval"})

yml = "config.yml"
subprocess.run(["rm", yml])

# --- training and validation data are always necessary!
delete = ["slurm", "cmd", "logger", "task", "dataset", "test_dataset", "val_dataset"]

generate_yml_config(checkpoint_path=checkpoint_path, yml=yml, delete=delete, update=update)

print(f"config yaml file seved to {yml}.")

t0 = time.time()
subprocess.run(f"python ./fairchem_main.py --mode train --config-yml {yml} --checkpoint {checkpoint_path}"
               f"&> train.txt", shell=True)

print(f"Elapsed time = {time.time() - t0:1.1f} seconds")

cpline  = subprocess.check_output(["grep", "checkpoint_dir", "train.txt"])
cpdir   = cpline.decode().strip().replace(" ", "").split(":")[-1]
newchk  = cpdir + "/checkpoint.pt"

print(f"new checkpoint: {os.path.abspath(newchk)}")
