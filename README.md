# Intracranial-CFD

A simple tool to perform vessel segmentation based on 3DRA and post contrast CBCT data.

## Usage

## Workflow
1. Batch conversion of DICOM to NIFTI format
2. Image registration (external process, recommend to use 3D slicer if manual initialization is needed)
3. Vessel segmentation with multiple Otsu thresholding
4. Extract surface as VTK file
5. Crop the defected vessel (with Paraview or other 3D mesh processing software)
6. Vessel centerline extraction
7. Multiplanar reconstruction (MPR) with input image and the centerline to straighten the image data
8. 