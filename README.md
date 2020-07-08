# Intracranial CFD

A tool collection to perform vessel segmentation based on 3DRA and post contrast CBCT intracranial data. The repository also provides an OpenFOAM computational fluid simulation pipeline.

## Workflow
1. Batch conversion of DICOM to NIFTI format
2. Image registration (external process, recommend to use 3D slicer if manual initialization is needed)
3. Located the defected point
4. Crop the image to smaller VOI
5. Vessel segmentation with multiple Otsu thresholding
6. Refinement on the segmented data (optional)
4. Extract surface as VTK file
6. Vessel centerline extraction
5. Crop the defected vessel (with Paraview or other 3D mesh processing software)
	1. Using fiducial point to locate the defected point
	2. Using `auto_vessel_seg.py` to automatically crop the defected vessel region
6. Extend the inlet and outlet region
7. Remeshing and volume meshing
8. CFD simulation
9. Probe CFD result along centerline
7. Multiplanar reconstruction (MPR) with input image and the centerline to straighten the image data

## Usage