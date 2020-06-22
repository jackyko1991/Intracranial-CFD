# OpenFOAM CFD Simulation on Vascular Structures

## Processing Workflow
1. After preparing vessel surface file (VTK/STL format), execute `batch_crop.py` for centerline extraction and inlet/outlet normalization (capping preferred)
2. Place the clipped surface at `<OpenFOAM-case-dir>/constant/triSurface`.
3. **Surface Feature Extraction:** (Optional) Edit `<OpenFOAM-case-dir>/system/surfaceFeatureExtractDict` and execute `surfaceFeatureExtractDict`.

4. **Create Background Mesh:** Generate a structured mesh which cells are divided into small cubes. Edit `<OpenFOAM-case-dir>/system/blockMeshDict` for the bounded domain and execute `blockMesh` to create background mesh. Files will be exported to `<OpenFOAM-case-dir>/constant/polyMesh`
5. **Volume Meshing:** 
	1. Edit `<OpenFOAM-case-dir>/system/snappyHexMeshDict` 
		1. The `snappyHexMeshDict` will control few mesh generation processes, including:
			- `castellatedMesh`: To create the first and rudimentary cube-based mesh. 
			- `snap`: Works on the cells at the walls to adapt their vertices to the initial geometry in order to obtain a smooth and realistic mesh surface
			- `addLayers`: Refine the mesh to compute an accurate wall shear stress. The geometry of mesh is not altered but add layers of cells at the walls of the vessel domain. This process reduces the distance between the first nodes, hence corrects surface imperfections.

	2. Execute `snappyHexMesh`

## Reference
- Hemodynamics in the Thoracic Aorta using OpenFOAM: 4D PCMRI versus CFD (https://core.ac.uk/download/pdf/81582466.pdf)
- OpenFOAM User Guide version 7 (http://foam.sourceforge.net/docs/Guides-a4/OpenFOAMUserGuide-A4.pdf)
- Online OpenFOAM UserGuide (https://www.openfoam.com/documentation/user-guide/index.php)