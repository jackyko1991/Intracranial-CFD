from tqdm import tqdm
import datetime
import os
from utils import *
import shutil
import tensorflow as tf

from simnet.solver import Solver
from simnet.dataset import TrainDomain, ValidationDomain, InferenceDomain, MonitorDomain
from simnet.data import BC, Validation, Inference, Monitor
from simnet.mesh_utils.mesh import Mesh
from simnet.PDES.navier_stokes import IntegralContinuity, NavierStokes
from simnet.controller import SimNetController
from simnet.csv_utils.csv_rw import csv_to_dict



class ICADSolver(Solver):

	def __init__(self, **config):
		super(ICADSolver, self).__init__(**config)

	@classmethod
	def update_defaults(cls, defaults):
		defaults.update({
			'network_dir': './network_checkpoint',
			'rec_results_cpu': True,
			'max_steps': 1500000,
			'decay_steps': 15000,
			})

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

	
	timepoints = range(1600,2100,100)
	openfoam_results = [os.path.join(case_dir,"CFD_OpenFOAM","VTK","OpenFOAM_" + str(timepoint) + ".vtk") for timepoint in timepoints]
	openfoam_result_csv_path = os.path.join(case_dir,"CFD_OpenFOAM_result","average.vtu")

	OpenFOAM_result_to_csv(openfoam_results,openfoam_result_csv_path)
	source_file = openfoam_result_csv_path
	target_file = "./domain/openfoam_result.csv"
	shutil.copy(source_file, target_file)

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

	# simnet controller
	# ctr = SimNetController(ICADSolver)