import os
import shutil
from multiprocessing import Pool
import datetime
import json
from tqdm import tqdm
import csv
from VascularSim import *
import time
from functools import partial

def worker(name,a=1):
	tqdm.write(str(a))
	tqdm.write(name)
	time.sleep(1)
	return name

def main():
	data_dir = "/mnt/DIIR-JK-NAS/data/intracranial"
	use_run_list = True

	sub_data_dirs = [
		"data_ESASIS_followup/medical",
		"data_ESASIS_followup/stent",
		"data_ESASIS_no_stenting",
		"data_surgery",
		"data_wingspan",
		"data_aneurysm_with_stenosis"
		]
		
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

	ensembled_case_list = []

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

				ensembled_case_list.append(os.path.join(data_dir,sub_data_dir,case,phase))

	# create thread pool
	pool_size = 5

	kwargs = {"output_vtk": True, "parallel":True, "cores":8, "cellNumber":20}
	mapFunc = partial(run_case, **kwargs)

	with Pool(pool_size) as p:
		r = list(tqdm(p.imap(mapFunc, ensembled_case_list), total=len(ensembled_case_list)))

if __name__ == "__main__":
	main()