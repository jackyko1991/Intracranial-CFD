import os
import SimpleITK as sitk
from tqdm import tqdm

def ExtractLargestConnectedComponents(label):
	ccFilter = sitk.ConnectedComponentImageFilter()
	label = ccFilter.Execute(label)

	labelStat = sitk.LabelShapeStatisticsImageFilter()
	labelStat.Execute(label)

	largestVol = 0
	largestVol2 = 0
	largestLabel = 0
	largestLabel2 = 0
	for labelNum in labelStat.GetLabels():
		if labelStat.GetPhysicalSize(labelNum) > largestVol:
			largestVol = labelStat.GetPhysicalSize(labelNum)
			largestLabel = labelNum
		elif labelStat.GetPhysicalSize(labelNum) > largestVol2:
			largestVol2 = labelStat.GetPhysicalSize(labelNum)
			largestLabel2 = labelNum
	print(largestLabel,largestLabel2)
	
	thresholdFilter = sitk.BinaryThresholdImageFilter()
	thresholdFilter.SetInsideValue(1)
	thresholdFilter.SetOutsideValue(0)

	thresholdFilter.SetLowerThreshold(largestLabel)
	thresholdFilter.SetUpperThreshold(largestLabel)
	label_0 = thresholdFilter.Execute(label)

	thresholdFilter.SetLowerThreshold(largestLabel2)
	thresholdFilter.SetUpperThreshold(largestLabel2)
	label_1 = thresholdFilter.Execute(label)

	addFilter = sitk.AddImageFilter()
	# label = addFilter.Execute(label_0,label_1)

	label = label_0

	return label

def preprocess(src,tgt):
	if not os.path.exists(src):
		return

	reader = sitk.ImageFileReader()
	reader.SetFileName(src)
	label = reader.Execute()

	# lcc
	label = ExtractLargestConnectedComponents(label)

	# smoothing
	# smoothFilter = sitk.BinaryMorphologicalClosingImageFilter()
	# smoothFilter = sitk.BinaryMorphologicalOpeningImageFilter()
	# smoothFilter.SetKernelType(sitk.sitkBall)
	# smoothFilter.SetKernelRadius(2)
	# label = smoothFilter.Execute(label)

	writer = sitk.ImageFileWriter()
	writer.SetFileName(tgt)
	writer.Execute(label)

def main():
	data_dir = "Z:/data/intracranial/followup_flatten/CBCT"
	pbar = tqdm(os.listdir(data_dir))

	cases = ["medical_LoCH_followup","medical_MokKP_followup"]

	for case in pbar:
		if not case in cases:
			continue
		pbar.set_description(case)
		preprocess(os.path.join(data_dir,case, "3DRA_seg_ICA_terminus.nii.gz"),os.path.join(data_dir,case, "CBCT_seg_ICA_terminus_lcc.nii.gz"))
		# exit()

if __name__ == "__main__":
	main()