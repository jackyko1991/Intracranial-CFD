from tqdm import tqdm
import datetime
import os
from utils import *
import shutil
import tensorflow as tf
import json
from sympy import Symbol, sqrt, Max
import vtk

from simnet.solver import Solver
from simnet.dataset import TrainDomain, ValidationDomain, InferenceDomain, MonitorDomain
from simnet.data import BC, Validation, Inference, Monitor
from simnet.mesh_utils.mesh import Mesh
from simnet.PDES.navier_stokes import IntegralContinuity, NavierStokes
from simnet.controller import SimNetController
from simnet.csv_utils.csv_rw import csv_to_dict

nu = 1
scale = 0.4
inlet_vel = 32.53 *1e-2

# inlet velocity profile
def circular_parabola(x, y, z, center, normal, radius, max_vel):
	centered_x = x-center[0]
	centered_y = y-center[1]
	centered_z = z-center[2]
	distance = sqrt(centered_x**2 + centered_y**2 + centered_z**2)
	parabola = max_vel*Max((1 - (distance/radius)**2), 0)
	return normal[0]*parabola, normal[1]*parabola, normal[2]*parabola

# normalize meshes
def normalize_mesh(mesh, center, scale):
	mesh.translate([-c for c in center])
	mesh.scale(scale)

class ICADTrain(TrainDomain):
	def __init__(self, **config):
		super(ICADTrain, self).__init__()
		needed_config = ICADTrain.process_config(config)
		self.__dict__.update(needed_config)

		# load domain json
		with open(config['config'].dataset_config) as f:
			domain = json.load(f)

		# get inlet outlet info
		outlets = []

		for key, value in domain.items():
			if "type" in value:
				if value["type"] == "inlet":
					inlet_stl = value["filename"]
					inlet_normal = value["tangent"]
					inlet_center = value["coordinate"]
					inlet_radius = value["radius"]
				elif value["type"] == "outlet":
					outlet_dict = {
						"stl":value["filename"],
						"normal": value["tangent"],
						"center": value["coordinate"],
						"radius": value["radius"]
					}
					outlets.append(outlet_dict)
		
		# read stl files to make meshes
		print("{}: Reading STL to mesh...".format(datetime.datetime.now()))
		inlet_mesh = Mesh.from_stl(os.path.join("stl_files",inlet_stl), airtight=False)
		print("{}: Reading inlet surface complete".format(datetime.datetime.now()))
		outlet_meshes = []

		for outlet in outlets:
			outlet_mesh = Mesh.from_stl(os.path.join("stl_files",outlet["stl"]), airtight=False)
			outlet_meshes.append(outlet_mesh)
		print("{}: Reading outlet surface complete".format(datetime.datetime.now()))

		noslip_mesh = Mesh.from_stl(os.path.join("stl_files",domain["vessel"]["filename"]), airtight=False)
		print("{}: Reading vessel surface complete".format(datetime.datetime.now()))

		# integral_mesh = Mesh.from_stl(point_path + 'aneurysm_integral.stl', airtight=False)
		interior_mesh = Mesh.from_stl(os.path.join("stl_files",domain["domain"]["filename"]))
		print("{}: Reading interior surface complete".format(datetime.datetime.now()))

		# normalize mesh
		center = (
			(inlet_mesh.bounds()["x"][0]+inlet_mesh.bounds()["x"][1])/2, 
			(inlet_mesh.bounds()["y"][0]+inlet_mesh.bounds()["y"][1])/2,
			(inlet_mesh.bounds()["z"][0]+inlet_mesh.bounds()["z"][1])/2
			)
		normalize_mesh(inlet_mesh, center, scale)
		normalize_mesh(outlet_mesh, center, scale)
		normalize_mesh(noslip_mesh, center, scale)
		# normalize_mesh(integral_mesh, center, scale)
		normalize_mesh(interior_mesh, center, scale)

		# Inlet
		u, v, w = circular_parabola(
		 	Symbol('x'),
		 	Symbol('y'),
			Symbol('z'),
			center=inlet_center,
			normal=inlet_normal,
			radius=inlet_radius,
			max_vel=inlet_vel
			)
		inlet = inlet_mesh.boundary_bc(outvar_sympy={'u': u, 'v': v, 'w': w},batch_size_per_area=256)
		self.add(inlet, name="Inlet")

		 # Outlet
		outlet = outlet_mesh.boundary_bc(outvar_sympy={'p': 0},batch_size_per_area=256)
		self.add(outlet, name="Outlet")

		# Noslip
		noslip = noslip_mesh.boundary_bc(outvar_sympy={'u': 0, 'v': 0, 'w': 0},batch_size_per_area=32)
		self.add(noslip, name="Noslip")

		# Interior
		interior = interior_mesh.interior_bc(outvar_sympy={
			'continuity': 0,
			'momentum_x': 0,
			'momentum_y': 0,
			'momentum_z': 0},
			batch_size_per_area=128,
			batch_per_epoch=1000)
		self.add(interior, name="Interior")

		# Integral Continuity 1
		ic_1 = outlet_mesh.boundary_bc(outvar_sympy={'integral_continuity': 2.540},
			lambda_sympy={'lambda_integral_continuity': 0.1},
			batch_size_per_area=128)
		self.add(ic_1, name="IntegralContinuity_1")

		# Integral Continuity 2
		ic_2 = inlet_mesh.boundary_bc(outvar_sympy={'integral_continuity': -2.540},
			lambda_sympy={'lambda_integral_continuity': 0.1},
			batch_size_per_area=128)
		self.add(ic_2, name="IntegralContinuity_2")

	@classmethod
	def add_options(cls,group):
		group.add_argument('--dataset_config',
			help='path to dataset config file',
			type=str,
			default='./dataset_config.json')

class ICADSolver(Solver):
	train_domain = ICADTrain
	# val_domain = ICADVal
	# monitor_domain = ICADMonitor

	def __init__(self, **config):
		super(ICADSolver, self).__init__(**config)
		needed_config = ICADSolver.process_config(config)
		self.__dict__.update(needed_config)

		self.equations = (NavierStokes(nu=nu*scale, rho=1, dim=3, time=False).make_node()
			+ IntegralContinuity(dim=3).make_node())
		flow_net = self.arch.make_node(name='flow_net',
			inputs=['x', 'y', 'z'],
			outputs=['u', 'v', 'w', 'p'])
		self.nets = [flow_net]

	@classmethod
	def update_defaults(cls, defaults):
		defaults.update({
			'dataset_config': './domain/domain.json',
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

	source_file = os.path.join(case_dir,"domain.json")
	target_file = "./domain/domain.json"
	shutil.copy(source_file, target_file)

	# timepoints = range(1600,2100,100)
	# openfoam_results = [os.path.join(case_dir,"CFD_OpenFOAM","VTK","OpenFOAM_" + str(timepoint) + ".vtk") for timepoint in timepoints]
	# openfoam_result_csv_path = os.path.join(case_dir,"CFD_OpenFOAM_result","average.vtu")

	# OpenFOAM_result_to_csv(openfoam_results,openfoam_result_csv_path)
	# source_file = openfoam_result_csv_path
	# target_file = "./domain/openfoam_result.csv"
	# shutil.copy(source_file, target_file)

	# stl surfaces
	with open("./domain/domain.json") as f:
		domain = json.load(f)

	stl_files = [domain["domain"]["filename"],domain["vessel"]["filename"]]

	for key, value in domain.items():
		if "type" in value:
			if value["type"] in ["inlet", "outlet"]:
				stl_files.append(value["filename"])

	for file in stl_files:
		if os.path.exists(os.path.join(case_dir,file)):
			source_file = os.path.join(case_dir,file)
			target_file = os.path.join("stl_files",file)
			shutil.copy(source_file,target_file)
		else:
			print("STL file not exists:{}".format(os.path.join(case_dir,file)))
			return

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

	# edit config for training

	# simnet controller
	ctr = SimNetController(ICADSolver)
	ctr.run()