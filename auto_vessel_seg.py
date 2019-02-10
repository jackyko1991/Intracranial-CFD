import os
import SimpleITK as sitk

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
	thresholdFilter.SetLowerThreshold(2) # 2 and/or 3, depends on data
	thresholdFilter.SetUpperThreshold(3.1)
	vessel_seg = thresholdFilter.Execute(seg)

	fillingFilter = sitk.BinaryFillholeImageFilter()
	vessel_seg = fillingFilter.Execute(vessel_seg)

	writer = sitk.ImageFileWriter()
	writer.SetFileName(vessel_path)
	writer.Execute(vessel_seg)

def main():
	data_folder = "D:/Cloud/Google Drive/intracranial vessels/followup/stent/LingKW/"

	# Extract3DRA(data_folder + "baseline/3DRA.nii",data_folder + "baseline/seg_vessel.nii")
	# Extract3DRA(data_folder + "baseline-post/3DRA.nii",data_folder + "baseline-post/seg_vessel.nii")
	# Extract3DRA(data_folder + "12months/3DRA.nii",data_folder + "12months/seg_vessel.nii")
	ExtractCBCT(data_folder + "followup/CBCT.nii",data_folder + "followup/seg_vessel.nii")

if __name__ == "__main__":
    main()