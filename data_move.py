import os
from tqdm import tqdm
import shutil

def forward():
	src_dir = "./by_case"
	tgt_dir = "./flatten"

	pbar = tqdm(os.listdir(src_dir))

	cases = ["WongPunCheong"]
	stages = ["pre","post"]
	phases = ["HA","PV"]

	# for case in pbar:
	for case in cases:
		pbar.set_description(case)
		for stage in stages:
			for phase in phases:
				# image
				if stage == "pre":
					src = os.path.join(src_dir,case,stage,"nii_reg",phase,"temporal_mIP.nii.gz")
				else:
					src = os.path.join(src_dir,case,stage,"nii_reg_pre",phase,"temporal_mIP.nii.gz")

				if not os.path.exists(src):
					continue
				tgt_dir_ = os.path.join(tgt_dir,case + "_" + stage + "_" + phase)
				tgt = os.path.join(tgt_dir_,"temporal_mIP.nii.gz")
				shutil.copy(src,tgt)

				# label
				src = os.path.join(src_dir,case,"label.nii.gz")

				if not os.path.exists(src):
					continue
				tgt_dir_ = os.path.join(tgt_dir,case + "_" + stage + "_" + phase)
				tgt = os.path.join(tgt_dir_,"label.nii.gz")

				os.makedirs(tgt_dir_,exist_ok=True)
				shutil.copy(src,tgt)

def backward():
	src_dir = "Z:/data/intracranial/followup_flatten"
	tgt_dir = "Z:/data/intracranial/followup"

	image_types = ["3DRA","CBCT"]
	filenames = []

	for image_type in image_types:
		pbar = tqdm(os.listdir(os.path.join(src_dir,image_type)))

		for folder in pbar:
			pbar.set_description(folder)
			split_foldername = folder.split("_")
			tx_type = case = split_foldername[0]
			case = split_foldername[1]
			stage = split_foldername[2]

			if image_type == "3DRA":
				filenames.append("3DRA_seg_ICA_terminus_lcc.nii.gz")
			elif image_type == "CBCT":
				filenames.append("CBCT_seg_ICA_terminus_lcc.nii.gz")

			for filename in filenames:
				src = os.path.join(src_dir,image_type,folder,filename)
				if not os.path.exists(src):
					continue

				tgt = os.path.join(tgt_dir,tx_type,case,stage,filename)
				shutil.copy(src,tgt)

def main():
	# forward()
	backward()


if __name__=="__main__":
	main()