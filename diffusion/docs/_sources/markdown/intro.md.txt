# Introduction
This library is made to calculate the proton diffusion constant
with ab-initio-based method.

## Proton diffusion in solid oxide
* In this example, the proton diffusion problem is investigated with NNP.
* The NNP provided by fairchem (prevoiusly OpenCatalyst Project) was used.
* The steps are as follows:
  1. Prepare the training data with DFT: VASP (not in this repository)
  2. Make LMDB file from DFT output: `make_lmdb_from_outcar.py`
  3. Prepare the pre-trained NNP model and do fine-tuning: `finetuning.py`
  4. Use fine-tuned NNP to the diffusion problem: `diffusion.py`

### 1. Prepare the training data
* The pre-trained NNP model is fine-tuned with the DFT program.
* To make the DFT-data, we perform the molecular dynamics calculation with VASP.
* The details of VAPS is not covered here so refer to the other repository.
* The OUTCAR file from the VASP calculation is taken. Here, the OUTCAR for BaZrO3 (the initial file for VASP
  is BaZrO3.cif) is prepared in the directory.

### 2. Make LMDB
* To do fine-tuning, we use LMDB database. This file is made by converting the OUTCAR file.
* The python file does the following:
  1. Load the OUTCAR file.
  2. Define the atom-to-graph object.
  3. Convert each step of OUTCAR to LMDB format.
  4. Copy the LMDB file to the data directory.
     Here, we use the same LMDB file for training and validation (which is not good ...)

### 3. Fine-tuning
* The python file does the following:
  1. Download the pre-trained NNP from the web.
  2. Define the fine-tuning configuration, and save it to yaml file ("config.yaml" here).
  3. Execute fine-tuning with "main.py".
  4. Save the result to checkpoint file ("checkpoint.pt").

### 4. Proton diffusion
* After having fine-tuned NNP, the real problem i.e. the proton diffusion in the solid oxide is carried out.
* To calculate the diffusion coefficient of proton (or hydrogen), we calculate the mean-square displacnement.
  We use the fact that the MSD should be fitted to the line, and the slope of the line is the diffusion coefficient (D).
* We can compare D with experiment, as this can be measured with experiment.
* The reference for the script: https://github.com/matlantis-pfcc/matlantis-contrib/blob/main/matlantis_contrib_examples/MD_Li_diffusion_in_LGPS/viewer.en.ipynb
* The python file does the following:
  1. Load the fine-tuned checkpoint file.
  2. Load the cif file and make the solid structure.
  3. Put the hydrogen atom (that we expect to diffuse).
  4. Perform the molecular dynamics calculation under constant volume and temperatue (NVT ensemble).
  5. Extract the position of hydrogen, and calculate the MSD.
  6. Fit the MSD to the line using *statmodels* library.
  7. Take slope of line and convert it to D.

