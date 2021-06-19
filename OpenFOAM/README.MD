# OpenFOAM CFD Simulation on Vascular Structures

This repository provides a fully automated CFD simulation pipeline using OpenFOAM. User should provide vessel STL surface with named region and associated centerline file for the process.

## Dependency installation  (Ubuntu 18.04)
1. Install OpenFOAM main package
2. We recommend users to use Anaconda 3.7 (https://www.anaconda.com/products/individual). Anaconda provide virtual environment for workspace dependency control. Follow the instruction to install all required packages:
	```bash
	$ conda create -n openfoam python=3.7
	$ conda activate openfoam
	$ pip install trimesh tqdm vtk pyfoam
	```

## Usage
### Single Run
Check `single_run.py -h` for the detail usage.

Quick example:
```bash
$ python single_run.py <case-dir> -v -p -c 8 -n 20
```
This will run in 8 cores with VTK output. `blockMesh` will perform with 20 cells in each direction.


### Batch Run
You have to provide `run_list.csv` along side with `all_run.py` within the same folder.

To change number of threads, edit `pool_size` to any integer > 1.

```bash
$ python all_run.py
```

## Processing Workflow
1. After preparing vessel surface file (VTK/STL format), execute `batch_crop.py` for centerline extraction and inlet/outlet normalization (capping preferred)
2. Place the clipped surface at `<OpenFOAM-case-dir>/constant/triSurface`.
3. **Surface Feature Extraction:** (Optional) Edit `<OpenFOAM-case-dir>/system/surfaceFeatureExtractDict` and execute `surfaceFeatureExtractDict`.

4. **Create Background Mesh:** Generate a structured mesh which cells are divided into small cubes. Edit `<OpenFOAM-case-dir>/system/blockMeshDict` for the bounded domain and execute `blockMesh` to create background mesh. Files will be exported to `<OpenFOAM-case-dir>/constant/polyMesh`
5. **Volume Meshing:** 
	1. Edit `<OpenFOAM-case-dir>/system/snappyHexMeshDict` 
		The `snappyHexMeshDict` will control few mesh generation processes, including:
		- `castellatedMesh`: To create the first and rudimentary cube-based mesh. (output as time point 1)
		- `snap`: Works on the cells at the walls to adapt their vertices to the initial geometry in order to obtain a smooth and realistic mesh surface (output as time point 2)
		- `addLayers`: Refine the mesh to compute an accurate wall shear stress. The geometry of mesh is not altered but add layers of cells at the walls of the vessel domain. This process reduces the distance between the first nodes, hence corrects surface imperfections. (output as time point 3)
	2. Execute `snappyHexMesh`. Use `snappyHexMesh -overwrite` to export final mesh only as time point 0.
6. **Finite Element Analysis:** Execute `pisoFoam` for CFD simulation.
7. **Output File (Optional):** 
	1. The `Allrun.py` batch processing file will create `OpenFOAM.OpenFOAM` file in the case directory for Paraview visualization. Users manually perform the CFD workflow can create the file by `touch OpenFOAM.OpenFOAM`. 
	2. The OpenFOAM result can be exported to VTK with command `foamToVTK`. Results will be exported to `<OpenFOAM-case-dir>/VTK`.
8. **Visualize the Result:** 
	1. Drag `OpenFOAM.OpenFOAM` to Paraview to visualize the result.
	2. STL file is usually in millimeter while OpenFOAM is executed in meter. To unify the scale, use `Filter`->`Transform`->`Scale`->(1000, 1000, 1000)

## Parallel Processing Benchmarking
The parallel processing test is on a 6C12T devices with different subdomain configurations. The procedure includes:
1. `blockMesh`
2. `surfaceFeatureExtract`
3. `decomposePar`
4. `snappyHexMesh`
5. `pisoFoam`
6. `reconstructParMesh`
7. `reconstructPar`.

Test results are as following:

| Cores         | Time(s)     | 
| ------------- |-------------| 
| 1             | 88.418      | 
| 2             | 60.699      | 
| 4             | 39.691      | 
| 8             | 51.170      |
|12             | 46.494      |

## Reference
- Hemodynamics in the Thoracic Aorta using OpenFOAM: 4D PCMRI versus CFD (https://core.ac.uk/download/pdf/81582466.pdf)
- OpenFOAM User Guide version 7 (http://foam.sourceforge.net/docs/Guides-a4/OpenFOAMUserGuide-A4.pdf)
- Online OpenFOAM UserGuide (https://www.openfoam.com/documentation/user-guide/index.php)
- OpenFOAM - open source CFD toolbox (https://events.prace-ri.eu/event/729/contributions/590/attachments/712/1015/Introduction_to_OpenFOAM_agrm_20180925.pdf)
- PyFoam (https://openfoamwiki.net/index.php/Contrib/PyFoam)

## Output
### Units
Density of blood = 1060kgm^-3
Pressure in mmHg = rho normalized pressure p* * density * 0.00750062mmHg/Pa = p* 7.95066 mmHg

- VTK file: 
	- velocity: ms-1 (after 1000x scaling)
	- pressure: 

- OpenFOAM file: 
	- velocity: ms^-1 (after 1000x scaling)
	- pressure: 1e-3 ms^-1