import SimpleITK as sitk
import os
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

def cbct_batch_smooth(src,tgt, LCC=True):
	reader = sitk.ImageFileReader()
	reader.SetFileName(src)
	label = reader.Execute()

	if LCC:
		label = ExtractLargestConnectedComponents(label)

	smoothFilter = sitk.BinaryMorphologicalClosingImageFilter()
	smoothFilter.SetKernelType(sitk.sitkBall)
	smoothFilter.SetKernelRadius(2)
	label = smoothFilter.Execute(label)

	writer = sitk.ImageFileWriter()
	writer.SetFileName(tgt)
	writer.Execute(label)

def main():
	kernelSize = 2

	LCC = True

	data_dir = "Z:/data/intracranial/followup_flatten/CBCT"
	src_filename = "3DRA_seg_ICA_terminus.nii.gz"

	if LCC:
		tgt_filename = "CBCT_seg_ICA_terminus_lcc.nii.gz"
	else:
		tgt_filename = "CBCT_seg_ICA_terminus.nii.gz"

	pbar = tqdm(os.listdir(data_dir))

	for case in pbar:
		pbar.set_description(case)

		src = os.path.join(data_dir,case,src_filename)
		tgt = os.path.join(data_dir,case,tgt_filename)

		if not os.path.exists(src):
			continue

		cbct_batch_smooth(src, tgt, LCC=LCC)

if __name__ == "__main__":
	main()