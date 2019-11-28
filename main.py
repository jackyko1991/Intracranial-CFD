import argparse
import dcm2nii
import os
import auto_vessel_seg

def parse_args():
	parser = argparse.ArgumentParser(description = "Intracranial Vessel Analysis Tool")
	parser.add_argument("--data-dir", type=str, default="../data/comparison/",
		help="Data directory")
	parser.add_argument("--dcm2nii", type=bool, default=False,
		help="Option to convert DICOM to Nifti")
	parser.add_argument("--segmentation", type=bool, default=True,
		help="Segment vessels")
	return parser.parse_args()

def batch_dcm2nii(args):
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
	print("Batch vessel segmentation...")

	dataDir = args.data_dir

	for patient in os.listdir(dataDir):
		if not os.path.exists(os.path.join(dataDir,patient,"3DRA","3DRA.nii")):
			continue
		auto_vessel_seg.Extract3DRA(os.path.join(dataDir,patient,"3DRA","3DRA.nii"),os.path.join(dataDir,patient,"3DRA","3DRA_seg.nii.gz"))
		auto_vessel_seg.LabelToSurface(os.path.join(dataDir,patient,"3DRA","3DRA_seg.nii.gz"),os.path.join(dataDir,patient,"3DRA","surface.vtk"))

		# if not os.path.exists(os.path.join(dataDir,patient,"CBCT","CBCT_reg.nii")):
		# 	continue
		# auto_vessel_seg.ExtractCBCT(os.path.join(dataDir,patient,"CBCT","CBCT_reg.nii"),os.path.join(dataDir,patient,"CBCT","CBCT_seg.nii.gz"))

		exit()

def main():
	args = parse_args()

	if args.dcm2nii:
		batch_dcm2nii(args)

	if args.segmentation:
		batch_segmentation(args)


if __name__=="__main__":
	main()