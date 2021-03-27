# SimNet Solver for Intracranial Stenosis

## Dependency
We strongly recommend to use conda virtual environment for version control. Python version only tested on 3.6. Check [dependency](./docs/SimNet_Install.md) for basic SimNet installation.

We will need following additional packages to fulfill the ICAD running environment. After switch to `simnet` conda virtual environment, type the command to install the packages:

```bash
pip install trimesh vtk tqdm
```

## Data Hierarchy
Place the dataset in arbitrary folder with following folder hierarchy


## Configuration File


## Training

Run the following command to start training
```bash
bash all_run.py
```

### Tensorboard
```bash
python all_run.py --run_mode=plot_data
```