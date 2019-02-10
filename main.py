import argparse
import dcm2nii
import os
import auto_vessel_seg

def parse_args():
	parser = argparse.ArgumentParser(description = "Intracranial Vessel Analysis Tool")
	parser.add_argument("--data-dir", type=str, default="../data/3DRA-CBCT/",
		help="Data directory")
	parser.add_argument("--dcm2nii", type=bool, default=True,
		help="Option to convert DICOM to Nifti")
	parser.add_argument("--segmentation", type=bool, default=False,
		help="Segment vessels")
	return parser.parse_args()

def batch_dcm2nii():
	print("Batch DICOM to Nifti conversion...")

	dataDir = args.data_dir
	for patient in os.listdir(dataDir):
		if not os.path.exists(os.path.join(dataDir,patient,"nii")):
			os.makedirs(os.path.join(dataDir,patient,"nii"))

		# 3DRA
		dcm2nii(os.path.join(dataDir,patient,"dicom","3DRA"),os.path.join(dataDir,patient,"nii","3DRA.nii.gz"))

		# CBCT
		dcm2nii(os.path.join(dataDir,patient,"dicom","CBCT"),os.path.join(dataDir,patient,"nii","CBCT.nii.gz"))

def batch_segmentation(args):
	print("Batch vessel segmentaion...")

	dataDir = args.data_dir

	# Extract3DRA(data_folder + "baseline/3DRA.nii",data_folder + "baseline/seg_vessel.nii")
	# Extract3DRA(data_folder + "baseline-post/3DRA.nii",data_folder + "baseline-post/seg_vessel.nii")
	# Extract3DRA(data_folder + "12months/3DRA.nii",data_folder + "12months/seg_vessel.nii")
	ExtractCBCT(data_folder + "followup/CBCT.nii",data_folder + "followup/seg_vessel.nii")

def main():
	args = parse_args()

	if args.dcm2nii:
		batch_dcm2nii(args)

	if args.segmentation:
		batch_segmentation(args)


if __name__=="__main__":
	main()