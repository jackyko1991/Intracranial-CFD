from tqdm import tqdm
import datetime
import os
from utils import *
import shutil

def run_case(case_dir, output_vtk=False):
	startTime = datetime.datetime.now()

	tqdm.write("********************************* SimNet CFD Operation *********************************")
	tqdm.write("{}: Execute SimNet CFD simulation on directory: {}".format(datetime.datetime.now(),case_dir))

	# tqdm.write("{}: STL domain merging...".format(datetime.datetime.now()))
	# stl_concat(os.path.join(case_dir,"domain.json"))

	# # copy surface from case directory
	# tqdm.write("{}: Copying necessary files...".format(datetime.datetime.now()))
	# source_file = os.path.join(case_dir,"domain_capped.stl")
	# target_file = "./stl_files/domain_capped.stl"
	# shutil.copy(source_file, target_file)

	# source_file = os.path.join(case_dir,"domain.json")
	# target_file = "./domain/domain.json"
	# shutil.copy(source_file, target_file)

	
	timepoints = range(1600,2000,100)
	openfoam_results = [os.path.join(case_dir,"CFD_OpenFOAM","VTK",str(timepoint)) for timepoint in timepoints]
	print(openfoam_results)

	# OpenFOAM_result_to_csv()

	# # clean workspace
	# tqdm.write("{}: Cleaning workspace...".format(datetime.datetime.now()))
	# if os.path.exists("./0/vorticity"):
	# 	os.remove("./0/vorticity")
	# if os.path.exists("./0/wallShearStress"):
	# 	os.remove("./0/wallShearStress")
	# if os.path.exists("./constant/polyMesh"):
	# 	shutil.rmtree("./constant/polyMesh")
	# if os.path.exists("./constant/extendedFeatureEdgeMesh"):
	# 	shutil.rmtree("./constant/extendedFeatureEdgeMesh")
	# for folder in os.listdir("./"):
	# 	try:
	# 		if folder == "0":
	# 			continue
	# 		is_cfd_result = float(folder)

	# 		shutil.rmtree(os.path.join("./",folder))
	# 	except ValueError:
	# 		continue