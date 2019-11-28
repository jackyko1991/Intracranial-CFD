import os
import SimpleITK as sitk
import vtk

def ExtractLargestConnectedComponents(label):
	ccFilter = sitk.ConnectedComponentImageFilter()
	label = ccFilter.Execute(label)

	labelStat = sitk.LabelShapeStatisticsImageFilter()
	labelStat.Execute(label)

	largestVol = 0
	largestLabel = 0
	for labelNum in labelStat.GetLabels():
		if labelStat.GetPhysicalSize(labelNum) > largestVol:
			largestVol = labelStat.GetPhysicalSize(labelNum)
			largestLabel = labelNum
	
	thresholdFilter = sitk.BinaryThresholdImageFilter()
	thresholdFilter.SetLowerThreshold(largestLabel)
	thresholdFilter.SetUpperThreshold(largestLabel)
	thresholdFilter.SetInsideValue(1)
	thresholdFilter.SetOutsideValue(0)
	label = thresholdFilter.Execute(label)

	return label

def Extract3DRA(img_path,vessel_path):
	print("Extracting vessel from 3DRA data:",img_path)
	reader = sitk.ImageFileReader()
	reader.SetFileName(img_path)
	img = reader.Execute()

	otsuFilter = sitk.OtsuMultipleThresholdsImageFilter()
	otsuFilter.SetNumberOfHistogramBins (256)
	otsuFilter.SetNumberOfThresholds(3)
	seg = otsuFilter.Execute(img)

	thresholdFilter = sitk.BinaryThresholdImageFilter()
	thresholdFilter.SetInsideValue (1)
	thresholdFilter.SetLowerThreshold(3)
	thresholdFilter.SetOutsideValue (0)
	thresholdFilter.SetUpperThreshold(4)
	vessel_seg = thresholdFilter.Execute(seg)

	vessel_seg = ExtractLargestConnectedComponents(vessel_seg)

	writer = sitk.ImageFileWriter()
	writer.SetFileName(vessel_path)
	writer.Execute(vessel_seg)

	# thresholdFilter = sitk.BinaryThresholdImageFilter()
	# thresholdFilter.SetInsideValue (1)
	# thresholdFilter.SetLowerThreshold(2)
	# thresholdFilter.SetOutsideValue (0)
	# thresholdFilter.SetUpperThreshold(2.1)
	# bone_seg = thresholdFilter.Execute(seg)

	# openingFilter = sitk.BinaryMorphologicalOpeningImageFilter()
	# openingFilter.SetKernelType(1)
	# openingFilter.SetKernelRadius(2)
	# bone_seg = openingFilter.Execute(bone_seg)

	# writer = sitk.ImageFileWriter()
	# writer.SetFileName(bone_path)
	# writer.Execute(bone_seg)

def ExtractCBCT(img_path,vessel_path):
	print("Extracting vessel from CBCT data:",img_path)
	reader = sitk.ImageFileReader()
	reader.SetFileName(img_path)
	img = reader.Execute()

	# reader.SetFileName(bone_mask_path)
	# bone_mask = reader.Execute()

	otsuFilter = sitk.OtsuMultipleThresholdsImageFilter()
	otsuFilter.SetNumberOfHistogramBins (256)
	otsuFilter.SetNumberOfThresholds(4)
	seg = otsuFilter.Execute(img)

	# thresholdFilter = sitk.ThresholdImageFilter()
	# # thresholdFilter.SetInsideValue (1)
	# thresholdFilter.SetLower(2) # 2 and/or 3, depends on data
	# thresholdFilter.SetUpper(3.1)
	# thresholdFilter.SetOutsideValue (0)
	# vessel_seg = thresholdFilter.Execute(seg)

	thresholdFilter = sitk.BinaryThresholdImageFilter()
	thresholdFilter.SetInsideValue (1)
	thresholdFilter.SetOutsideValue (0)
	thresholdFilter.SetLowerThreshold(3) # 2 and/or 3, depends on data
	thresholdFilter.SetUpperThreshold(3.1)
	vessel_seg = thresholdFilter.Execute(seg)

	fillingFilter = sitk.BinaryFillholeImageFilter()
	vessel_seg = fillingFilter.Execute(vessel_seg)

	vessel_seg = ExtractLargestConnectedComponents(vessel_seg)

	writer = sitk.ImageFileWriter()
	writer.SetFileName(vessel_path)
	writer.Execute(vessel_seg)

def LabelToSurface(label_path, surface_path):
	print("Converting segmentation label to surface file:",label_path)
	reader = sitk.ImageFileReader()
	reader.SetFileName(label_path)
	label = reader.Execute()

	# binary path
	bin_path = "D:/Dr_Simon_Yu/CFD_intracranial/code/cxx/label_to_mesh/build/Release/LabelToMesh.exe"

	# convert to binary file
	command = bin_path + " " + label_path + " " + surface_path + " 1 1"
	# print(command)
	os.system(command)

	reader = vtk.vtkGenericDataObjectReader()
	reader.SetFileName(surface_path)
	reader.Update()
	surface = reader.GetOutput()

	transform = vtk.vtkTransform()
	transform.Scale(-1,-1,1)

	transformFilter = vtk.vtkTransformPolyDataFilter()
	transformFilter.SetInputData(surface)
	transformFilter.SetTransform(transform)
	transformFilter.Update()
	surface = transformFilter.GetOutput()

	smoothFilter = vtk.vtkSmoothPolyDataFilter()
	smoothFilter.SetInputData(surface)
	smoothFilter.SetNumberOfIterations(15);
	smoothFilter.SetRelaxationFactor(0.1);
	smoothFilter.FeatureEdgeSmoothingOff();
	smoothFilter.BoundarySmoothingOn();
	smoothFilter.Update();
	surface = smoothFilter.GetOutput()

	writer = vtk.vtkGenericDataObjectWriter()
	writer.SetFileName(surface_path)
	writer.SetInputData(surface)
	writer.Update()

def main():
	data_folder = "D:/Dr_Simon_Yu/CFD_intracranial/data/comparison/BlasiRaquelLegaspi/3DRA/"

	# Extract3DRA(data_folder + "baseline/3DRA.nii",data_folder + "baseline/seg_vessel.nii")
	# Extract3DRA(data_folder + "baseline-post/3DRA.nii",data_folder + "baseline-post/seg_vessel.nii")
	# Extract3DRA(data_folder + "12months/3DRA.nii",data_folder + "12months/seg_vessel.nii")
	# ExtractCBCT(data_folder + "followup/CBCT.nii",data_folder + "followup/seg_vessel.nii")
	LabelToSurface(os.path.join(data_folder,"3DRA_seg.nii.gz"), os.path.join(data_folder,"surface.vtk"))

if __name__ == "__main__":
    main()