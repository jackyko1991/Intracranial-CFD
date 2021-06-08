import os
import shutil
from PyFoam.RunDictionary.ParsedBlockMeshDict import ParsedBlockMeshDict
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
import vtk
import datetime
import json
from tqdm import tqdm
import trimesh
import csv
from VascularSim import *

def main():
	data_dir = "/mnt/DIIR-JK-NAS/data/intracranial"
	use_run_list = True

	sub_data_dirs = [
		"data_ESASIS_followup/medical",
		"data_ESASIS_followup/stent",
		"data_ESASIS_no_stenting",
		#"data_surgery",
		"data_wingspan",
		#"data_aneurysm_with_stenosis"
		]

	# data_dir = "/mnt/DIIR-JK-NAS/data/intracranial/data_30_30"
	# sub_data_dirs = ["surgery"]

	# phases = ["baseline", "baseline-post", "12months", "followup"]
	phases = ["baseline"]

	if use_run_list:
		run_list = "./run_list.csv"
		with open(run_list, newline='') as csvfile:
			reader = csv.reader(csvfile)
			run_cases = [l[0] for l in reader]

	ignore_cases = [
		"002",
		"057",
		]

	for sub_data_dir in sub_data_dirs:
		datalist = os.listdir(os.path.join(data_dir,sub_data_dir))
		pbar = tqdm(datalist)

		for case in pbar:
			pbar.set_description(case)

			if use_run_list:
				if not case in run_cases:
					continue

			if case in ignore_cases:
				continue

			for phase in phases:
				if not os.path.exists(os.path.join(data_dir,sub_data_dir,case,phase)):
					continue

				if not os.path.exists(os.path.join(data_dir,sub_data_dir,case,phase,"domain.json")):
					continue

				#if os.path.exists(os.path.join(data_dir,sub_data_dir,case,phase,"CFD_OpenFOAM")):
				#	continue

				run_case(os.path.join(data_dir,sub_data_dir,case,phase),output_vtk=True, parallel=True, cores=8)

			# run_case(os.path.join(data_dir,sub_data_dir,case),output_vtk=True, parallel=True, cores=8)

if __name__ == "__main__":
	main()