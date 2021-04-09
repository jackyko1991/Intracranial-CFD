import os
from icad_solver import *
from tqdm import tqdm

def main():
	data_dir = "/mnt/DIIR-JK-NAS/data/intracranial"
	# data_dir = "Z:/data/intracranial"
	sub_data_dirs = [
		# "data_ESASIS_followup/medical",
		# "data_ESASIS_followup/stent",
		# "data_ESASIS_no_stenting",
		# "data_surgery",
		# "data_wingspan",
		"data_sample"
		]

	phases = ["baseline"]

	for sub_data_dir in sub_data_dirs:
		datalist = os.listdir(os.path.join(data_dir,sub_data_dir))
		
		pbar = tqdm(datalist)

		for case in pbar:
			pbar.set_description(case)

			for phase in phases:
				if not os.path.exists(os.path.join(data_dir,sub_data_dir,case,phase)):
					continue

				if not os.path.exists(os.path.join(data_dir,sub_data_dir,case,phase,"domain.json")):
					continue

				#if os.path.exists(os.path.join(data_dir,sub_data_dir,case,phase,"CFD_OpenFOAM")):
				#	continue

				run_case(os.path.join(data_dir,sub_data_dir,case,phase),output_vtk=True)
				exit()

if __name__ == "__main__":
	main()