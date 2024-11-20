from __future__ import annotations

import os
import shutil
import warnings

import numpy as np
import pytorch_lightning as pl
from functools import partial
from dgl.data.utils import split_dataset
from mp_api.client import MPRester
from pytorch_lightning.loggers import CSVLogger

import matgl
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_pes
from matgl.models import M3GNet
from matgl.utils.training import PotentialLightningModule
from matgl.config import DEFAULT_ELEMENTS


m3gnet_nnp = matgl.load_model("M3GNet-MP-2021.2.8-PES")
model_pretrained = m3gnet_nnp.model
lit_module_finetune = PotentialLightningModule(model=model_pretrained, lr=1e-4, include_line_graph=True)
