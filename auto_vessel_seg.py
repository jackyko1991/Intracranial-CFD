import os
import SimpleITK as sitk
import vtk

def Extract3DRA(img_path,vessel_path):
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
	thresholdFilter.SetUpperThreshold(3.1)
	vessel_seg = thresholdFilter.Execute(seg)

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
	reader = sitk.ImageFileReader()
	reader.SetFileName(img_path)
	img = reader.Execute()

	reader.SetFileName(bone_mask_path)
	bone_mask = reader.Execute()

	otsuFilter = sitk.OtsuMultipleThresholdsImageFilter()
	otsuFilter.SetNumberOfHistogramBins (256)
	otsuFilter.SetNumberOfThresholds(4)
	seg = otsuFilter.Execute(img)

	thresholdFilter = sitk.BinaryThresholdImageFilter()
	thresholdFilter.SetInsideValue (1)
	thresholdFilter.SetLowerThreshold(2)
	thresholdFilter.SetOutsideValue (0)
	thresholdFilter.SetUpperThreshold(2.1)
	vessel_seg = thresholdFilter.Execute(seg)

	fillingFilter = sitk.BinaryFillholeImageFilter()
	vessel_seg = fillingFilter.Execute(vessel_seg)

	writer = sitk.ImageFileWriter()
	writer.SetFileName(vessel_path)
	writer.Execute(vessel_seg)

def main():
	data_folder = "I:/CFD/intracranial CBCT 3DRA/comparison/ChickFK/"

	Extract3DRA(data_folder + "3DRA/3DRA.nii",data_folder + "3DRA/seg_vessel.nii")
	ExtractCBCT(data_folder + "CBCT/CBCT_resample.nii",data_folder + "CBCT/seg_vessel.nii")

if __name__ == "__main__":
    main()