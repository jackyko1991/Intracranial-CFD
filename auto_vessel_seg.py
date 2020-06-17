import os
import SimpleITK as sitk
import vtk
from tqdm import tqdm

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

def LabelToSurface(label_path, surface_path, lcc=False, smoothIterations=15,relaxationFactor=0.1):
	print("Converting segmentation label to surface file:",label_path)

	# binary path
	bin_path = "D:/projects/CFD_intracranial/cxx/label_to_mesh/build/Release/LabelToMesh.exe"

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
	smoothFilter.SetNumberOfIterations(smoothIterations);
	smoothFilter.SetRelaxationFactor(relaxationFactor);
	smoothFilter.FeatureEdgeSmoothingOff();
	smoothFilter.BoundarySmoothingOn();
	smoothFilter.Update();
	surface = smoothFilter.GetOutput()

	if lcc:
		connectedFilter = vtk.vtkConnectivityFilter()
		connectedFilter.SetInputData(surface)
		connectedFilter.Update()
		surface = connectedFilter.GetOutput()

	writer = vtk.vtkGenericDataObjectWriter()
	writer.SetFileName(surface_path)
	writer.SetInputData(surface)
	writer.Update()

def crop_defected_region(image_path, defected_point_csv_path, cropped_image_path,crop_size=[50,50,50]):
	reader = sitk.ImageFileReader()
	reader.SetFileName(image_path)
	image = reader.Execute()

	defected_point = open(defected_point_csv_path,"r").readlines()[3]
	defected_point = defected_point.split(",")[1:4]
	defected_point = [float(i) for i in defected_point]
	defected_point[0] = -1*defected_point[0]
	defected_point[1] = -1*defected_point[1]

	defected_index = image.TransformPhysicalPointToIndex(defected_point)
	start_point = [defected_point[i]-crop_size[i]/2 for i in range(3)]
	end_point = [defected_point[i]+crop_size[i]/2 for i in range(3)]
	
	start_indices = list(image.TransformPhysicalPointToIndex(start_point))
	end_indices = list(image.TransformPhysicalPointToIndex(end_point))
	
	new_size = []
	for i in range(3):
		if start_indices[i] < 0:
			start_indices[i] = 0
		if start_indices[i] >= image.GetSize()[i]:
			start_indices[i] = image.GetSize()[i]-1

		if end_indices[i] < 0:
			end_indices[i] = 0
		if end_indices[i] >= image.GetSize()[i]:
			end_indices[i] = image.GetSize()[i]-1

		size_ = abs(end_indices[i]-start_indices[i])

		if start_indices[i] > end_indices[i]:
			tmp = start_indices[i]
			start_indices[i] = end_indices[i]
			end_indices[i] = tmp

		new_size.append(size_)

	roiFilter = sitk.RegionOfInterestImageFilter()
	roiFilter.SetSize(new_size)
	roiFilter.SetIndex(start_indices)
	image = roiFilter.Execute(image)

	writer = sitk.ImageFileWriter()
	writer.SetFileName(cropped_image_path)
	writer.Execute(image)

def main():
	data_folder = "Z:/data/intracranial/followup"
	# tx_types = ["medical","stent"]
	tx_types = ["medical"]
	phases = ["baseline","baseline-post","12months","followup"]
	label_filename = "3DRA_seg_ICA_terminus.nii.gz"
	lcc_label_filename_3DRA = "3DRA_seg_ICA_terminus_lcc.nii.gz"
	lcc_label_filename_CBCT = "CBCT_seg_ICA_terminus_lcc.nii.gz"

	output_surface_name = "surface.vtk"

	for tx_type in tx_types:
		pbar = tqdm(os.listdir(os.path.join(data_folder,tx_type)))

		for case in pbar:
			pbar.set_description(case)

			for phase in phases:
				if phase == "followup":
					label_file = os.path.join(data_folder,tx_type,case, phase, lcc_label_filename_CBCT)
				else:
					label_file = os.path.join(data_folder,tx_type,case, phase, lcc_label_filename_3DRA)
				if not os.path.exists(label_file):
					continue
				output_surface = os.path.join(data_folder,tx_type,case, phase, output_surface_name)

				# Extract3DRA(data_folder + "baseline/3DRA.nii",data_folder + "baseline/seg_vessel.nii")
				# Extract3DRA(data_folder + "baseline-post/3DRA.nii",data_folder + "baseline-post/seg_vessel.nii")
				# Extract3DRA(data_folder + "12months/3DRA.nii",data_folder + "12months/seg_vessel.nii")
				# ExtractCBCT(data_folder + "followup/CBCT.nii",data_folder + "followup/seg_vessel.nii")
				if phase=="followup":
					LabelToSurface(label_file, output_surface,relaxationFactor=0.5)
				else:
					LabelToSurface(label_file, output_surface,relaxationFactor=0.1)
				# crop_defected_region(os.path.join(data_folder,"3DRA","3DRA.nii"), os.path.join(data_folder,"defected_point.fcsv"),os.path.join(data_folder,"3DRA","3DRA_cropped.nii.gz"))
				# crop_defected_region(os.path.join(data_folder,"3DRA","3DRA_seg.nii.gz"), os.path.join(data_folder,"defected_point.fcsv"),os.path.join(data_folder,"3DRA","3DRA_seg_cropped.nii.gz"))
				# crop_defected_region(os.path.join(data_folder,"CBCT","CBCT_reg.nii"), os.path.join(data_folder,"defected_point.fcsv"),os.path.join(data_folder,"CBCT","CBCT_reg_cropped.nii.gz"))

if __name__ == "__main__":
    main()