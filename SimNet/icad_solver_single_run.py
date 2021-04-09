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

# nu = 0.025
rho = 1
nu = 3.365*1e-6
# rho = 1040
scale = 0.4
# scale = 0.1
inlet_vel = 32.53 *1e-2
# inlet_vel = 1.5
continuity_unit = 2.5
mapping = {'Points:0': 'x', 'Points:1': 'y', 'Points:2': 'z', 'U_average:0': 'u', 'U_average:1': 'v', 'U_average:2': 'w', 'p_average': 'p'}
continuity = False
inlet_profile = "constant" #constant, parabolic
batch_size_per_area_boundary = 256

# inlet velocity profile
def circular_parabola(x, y, z, center, normal, radius, max_vel):
	centered_x = x-center[0]
	centered_y = y-center[1]
	centered_z = z-center[2]

	distance = sqrt(centered_x**2 + centered_y**2 + centered_z**2)
	parabola = max_vel*Max((1 - (distance/radius)**2), 0)

	return normal[0]*parabola, normal[1]*parabola, normal[2]*parabola

def circular_constant(x, y, z, center, normal, radius,vel):
	centered_x = x-center[0]
	centered_y = y-center[1]
	centered_z = z-center[2]

	distance = sqrt(centered_x**2 + centered_y**2 + centered_z**2)
	circualr = vel*Max(distance/radius,0.9999)
	return normal[0]*circualr, normal[1]*circualr, normal[2]*circualr

# normalize meshes
def normalize_mesh(mesh, center, scale):
	mesh.translate([-c for c in center])
	mesh.scale(scale)

# normalize invars
def normalize_invar(invar, center, scale, dims=2):
	invar['x'] -= center[0]
	invar['y'] -= center[1]
	invar['z'] -= center[2]
	invar['x'] *= scale
	invar['y'] *= scale
	invar['z'] *= scale
	if 'area' in invar.keys():
		invar['area'] *= scale**dims
	return invar

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

		if continuity:
			integral_mesh = Mesh.from_stl(os.path.join("stl_files","domain_integral.stl"), airtight=False)
			print("{}: Reading integral domain complete".format(datetime.datetime.now()))

		interior_mesh = Mesh.from_stl(os.path.join("stl_files",domain["domain"]["filename"]))
		print("{}: Reading interior surface complete".format(datetime.datetime.now()))

		# normalize mesh
		center = (
			(interior_mesh.bounds()["x"][0]+interior_mesh.bounds()["x"][1])/2, 
			(interior_mesh.bounds()["y"][0]+interior_mesh.bounds()["y"][1])/2,
			(interior_mesh.bounds()["z"][0]+interior_mesh.bounds()["z"][1])/2
			)
		print("Mesh center: ({:.4f}, {:.4f}, {:.4f})".format(center[0],center[1],center[2]))
		normalize_mesh(inlet_mesh, center, scale)

		for outlet_mesh in outlet_meshes:
			normalize_mesh(outlet_mesh, center, scale)
		normalize_mesh(noslip_mesh, center, scale)
		if continuity:
			normalize_mesh(integral_mesh, center, scale)
		normalize_mesh(interior_mesh, center, scale)

		inlet_center = ((inlet_center[0]-center[0])*scale,(inlet_center[1]-center[1])*scale,(inlet_center[2]-center[2])*scale)

		print("inlet center: ",inlet_center)

		# Inlet
		if inlet_profile == "parabolic":
			u, v, w = circular_parabola(
			 	Symbol('x'),
			 	Symbol('y'),
				Symbol('z'),
				center=inlet_center,
				normal=inlet_normal,
				radius=inlet_radius*scale,
				max_vel=inlet_vel
				)
		elif inlet_profile == "constant":
			u, v, w = circular_constant(
			 	Symbol('x'),
			 	Symbol('y'),
				Symbol('z'),
				center=inlet_center,
				normal=inlet_normal,
				radius=inlet_radius*scale,
				vel=inlet_vel
				)
		inlet = inlet_mesh.boundary_bc(outvar_sympy={'u': u, 'v': v, 'w': w},batch_size_per_area=batch_size_per_area_boundary)
		self.add(inlet, name="Inlet")

		# Outlet
		for i, outlet_mesh in enumerate(outlet_meshes):
			outlet = outlet_mesh.boundary_bc(outvar_sympy={'p': 0},batch_size_per_area=batch_size_per_area_boundary)
			self.add(outlet, name="Outlet_" + str(i))

		# Noslip
		noslip = noslip_mesh.boundary_bc(outvar_sympy={'u': 0, 'v': 0, 'w': 0}, batch_size_per_area=32)
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

		if continuity:
			# # Integral Continuity for outlets
			# for i, outlet_mesh in enumerate(outlet_meshes):
			# 	## here I tried outlet to inlet area ratio for continuity
			# 	ic = outlet_mesh.boundary_bc(outvar_sympy={'integral_continuity': continuity_unit*(outlets[i]["radius"]/inlet_radius)**2},
			# 		lambda_sympy={'lambda_integral_continuity': 0.1},
			# 		batch_size_per_area=128)
			# 	self.add(ic, name="IntegralContinuity_" + str(i))

			# Integral Continuity for inlet
			ic_inlet = inlet_mesh.boundary_bc(outvar_sympy={'integral_continuity': -continuity_unit},
				lambda_sympy={'lambda_integral_continuity': 0.1},
				batch_size_per_area=128)
			self.add(ic_inlet, name="IntegralContinuity_inlet")

			# Integral Continuity for integral domain
			# plane direction correct?
			ic_integral = integral_mesh.boundary_bc(outvar_sympy={'integral_continuity': -continuity_unit},
				lambda_sympy={'lambda_integral_continuity': 0.1},
				batch_size_per_area=128)
			self.add(ic_integral, name="IntegralContinuity_custom")

	@classmethod
	def add_options(cls,group):
		group.add_argument('--dataset_config',
			help='path to dataset config file',
			type=str,
			default='./dataset_config.json')

class ICADVal(ValidationDomain):
	def __init__(self, **config):
		super(ICADVal, self).__init__()
		needed_config = ICADTrain.process_config(config)
		self.__dict__.update(needed_config)

		# load domain json
		with open(config['config'].dataset_config) as f:
			domain = json.load(f)

		# read validation data
		print("{}: Loading validation CSV file...".format(datetime.datetime.now()))
		openfoam_var = csv_to_dict('./openfoam/openfoam_result.csv', mapping)
		print("{}: Loading validation CSV file complete".format(datetime.datetime.now()))
		openfoam_invar = {key: value for key, value in openfoam_var.items() if key in ['x', 'y', 'z']}
		openfoam_outvar = {key: value for key, value in openfoam_var.items() if key in ['u', 'v', 'w', 'p']}

		# normalize openfoam result
		interior_mesh = Mesh.from_stl(os.path.join("stl_files",domain["domain"]["filename"]))
		print("{}: Reading interior surface complete".format(datetime.datetime.now()))

		# normalize mesh
		center = (
			(interior_mesh.bounds()["x"][0]+interior_mesh.bounds()["x"][1])/2, 
			(interior_mesh.bounds()["y"][0]+interior_mesh.bounds()["y"][1])/2,
			(interior_mesh.bounds()["z"][0]+interior_mesh.bounds()["z"][1])/2
			)

		openfoam_invar = normalize_invar(openfoam_invar, center, scale, dims=3)

		val = Validation.from_numpy(openfoam_invar, openfoam_outvar)
		self.add(val, name='Val')

# class ICADMonitor(MonitorDomain):
# 	def __init__(self, **config):
# 		super(ICADMonitor, self).__init__()
# 		# metric for pressure drop
# 		metric = Monitor(inlet_mesh.sample_boundary(16),{'pressure_drop': lambda var: tf.reduce_mean(var['p'])})
# 		self.add(metric, 'PressureDrop')

class ICADSolver(Solver):
	train_domain = ICADTrain
	val_domain = ICADVal
	# monitor_domain = ICADMonitor

	def __init__(self, **config):
		super(ICADSolver, self).__init__(**config)
		needed_config = ICADSolver.process_config(config)
		self.__dict__.update(needed_config)

		if continuity:
			self.equations = (NavierStokes(nu=nu*scale, rho=rho, dim=3, time=False).make_node()
				+ IntegralContinuity(dim=3).make_node())
		else:
			self.equations = NavierStokes(nu=nu*scale, rho=rho, dim=3, time=False).make_node()
		flow_net = self.arch.make_node(name='flow_net',
			inputs=['x', 'y', 'z'],
			outputs=['u', 'v', 'w', 'p'])
		self.nets = [flow_net]

	@classmethod
	def update_defaults(cls, defaults):
		defaults.update({
			'dataset_config': './domain/domain.json',
			'network_dir': './network_checkpoint_segment',
			# 'start_lr': 1e-2,
			# 'rec_results_freq': 100,
			'rec_results_cpu': True,
			'max_steps': 1500000,
			'decay_steps': 15000,
			})

if __name__=="__main__":
	# simnet controller
	ctr = SimNetController(ICADSolver)
	ctr.run()