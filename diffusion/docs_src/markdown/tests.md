# Test cases

1. BaZrO3
* BaZrO3 is a popular proton conducting material.
* Calculation is done in the following order.
  1. Prepare the training data with DFT: VASP (not in this repository)
  2. Make LMDB file from DFT output: `make_lmdb_from_outcar.py`
  3. Prepare the pre-trained NNP model and do fine-tuning: `finetuning.py`
  4. Use fine-tuned NNP to the diffusion problem: `diffusion.py`
