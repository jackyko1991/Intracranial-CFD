import os
import shutil
from PyFoam.RunDictionary.ParsedBlockMeshDict import ParsedBlockMeshDict
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
import vtk
import datetime
import json
from tqdm import tqdm
import trimesh

def edit_blockMeshDict(dictionary, stl, edge_buffer=2):
	try:
		tqdm.write("{}: Editing blockMeshDict: {}".format(datetime.datetime.now(),dictionary))
		blockMeshDict = ParsedBlockMeshDict(dictionary)
		vertices = blockMeshDict["vertices"]

		# load stl surface
		reader = vtk.vtkSTLReader()
		reader.SetFileName(stl)
		reader.Update()
		bounds = reader.GetOutput().GetBounds()

		blockMeshDict["xmin"] =  bounds[0]-edge_buffer;
		blockMeshDict["xmax"] =  bounds[1]+edge_buffer;
		blockMeshDict["ymin"] =  bounds[2]-edge_buffer;
		blockMeshDict["ymax"] =  bounds[3]+edge_buffer;
		blockMeshDict["zmin"] =  bounds[4]-edge_buffer;
		blockMeshDict["zmax"] =  bounds[5]+edge_buffer;

		cellNumber = 40

		blocks = blockMeshDict["blocks"]
		blocks[2] = '({} {} {})'.format(cellNumber,cellNumber,cellNumber)
		blockMeshDict["blocks"] = blocks

		try:
			blockMeshDict.writeFile()
		except IOError:
			tqdm.write("Can't write file. Content would have been:")
			tqdm.write(blockMeshDict)

		return 0

	except IOError:
		tqdm.write(blockMeshDict_file, "does not exist")
		return 1

def edit_decompseParDict(dictionary, cores=4):
	try:
		tqdm.write("{}: Editing decompseParDict: {}".format(datetime.datetime.now(),dictionary))
		decompseParDict = ParsedParameterFile(dictionary)

		available_core_config = [2,4,8,12]
		if cores not in available_core_config:
			raise ValueError("Invalid core numbers, possible choices: (2, 4, 8, 12)")

		decompseParDict["numberOfSubdomains"] = cores
		if decompseParDict["numberOfSubdomains"] == 2:
			simpleCoeffs = {"n": str("(2 1 1)"), "delta": 0.001}
			hierarchicalCoeffs = {"n": str("(2 1 1)"), "delta": 0.001, "order": "xyz"}
		elif decompseParDict["numberOfSubdomains"] == 4:
			simpleCoeffs = {"n": str("(2 2 1)"), "delta": 0.001}
			hierarchicalCoeffs = {"n": str("(2 2 1)"), "delta": 0.001, "order": "xyz"}
		elif decompseParDict["numberOfSubdomains"] == 8:
			simpleCoeffs = {"n": str("(2 2 2)"), "delta": 0.001}
			hierarchicalCoeffs = {"n": str("(2 2 2)"), "delta": 0.001, "order": "xyz"}
		elif decompseParDict["numberOfSubdomains"] == 12:
			simpleCoeffs = {"n": str("(3 2 2)"), "delta": 0.001}
			hierarchicalCoeffs = {"n": str("(3 2 2)"), "delta": 0.001, "order": "xyz"}

		decompseParDict["simpleCoeffs"] = simpleCoeffs
		decompseParDict["hierarchicalCoeffs"] = hierarchicalCoeffs

		try:
			decompseParDict.writeFile()
		except IOError:
			tqdm.write("Can't write file. Content would have been:")
			tqdm.write(decompseParDict)

	except IOError:
		tqdm.write(dictionary, "does not exist")
		return 1	

def edit_velocity(dictionary, domain_json, velocity=1):
	"""
	Edit the initial velocity file for CFD simulation

	:param str dictionary: File path of the velocity dictionary, (0/U)
	:param str domain_json: File path of vessel inlet description json
	:param float velocity: Inlet velocity magnitude (cm/s)
	"""

	try:
		tqdm.write("{}: Editing velocity file: {}".format(datetime.datetime.now(),dictionary))
		velocityDict = ParsedParameterFile(dictionary)

		with open(domain_json, 'r') as f:
			domain_dict = json.load(f)

		boundaryField = {"vessel": {"type": "noSlip"}, "walls":{"type":"empty"}}	
		for key, value in domain_dict.items():
			try:
				if value["type"] == "inlet":
					boundaryField.update({key: {
						"type": "fixedValue",
						"value": "uniform (" +\
							str(value["tangent"][0]/100*velocity) + " " + \
							str(value["tangent"][1]/100*velocity) + " " + \
							str(value["tangent"][2]/100*velocity) + ")"}})
				elif value["type"] == "outlet":
					boundaryField.update({key: { 
						"type": "inletOutlet",
						"inletValue": "uniform (0 0 0)",
						"value":"uniform (0 0 0)"}}
						)
				else:
					continue
			except:
				continue

		velocityDict["boundaryField"] = boundaryField

		try:
			velocityDict.writeFile()
		except IOError:
			tqdm.write("Can't write file. Content would have been:")
			tqdm.write(velocityDict)

	except IOError:
		tqdm.write(dictionary, "does not exist")
		return 1

def edit_pressure(dictionary, domian_json, pressure=0):
	"""
	Edit the initial pressure file for CFD simulation

	:param str dictionary: File path of the velocity dictionary, (0/U)
	:param str domian_json: File path of vessel inlet description json
	:param float pressure: Outlet pressure
	"""

	try:
		tqdm.write("{}: Editing pressure file: {}".format(datetime.datetime.now(),dictionary))
		pressureDict = ParsedParameterFile(dictionary)

		with open(domian_json, 'r') as f:
			domain_dict = json.load(f)

		boundaryField = {"vessel": {"type": "zeroGradient"}, "walls":{"type":"empty"}}	
		for key, value in domain_dict.items():
			try:			
				if value["type"] == "inlet":
					boundaryField.update({key: { 
						"type": "zeroGradient"}})
				elif value["type"] == "outlet":
					boundaryField.update({key: { 
						"type": "fixedValue",
						"value":"uniform 0 "}}
						)
				else:
					continue
			except:
				continue

		pressureDict["boundaryField"] = boundaryField

		try:
			pressureDict.writeFile()
		except IOError:
			tqdm.write("Can't write file. Content would have been:")
			tqdm.write(pressureDict)

	except IOError:
		tqdm.write(dictionary, "does not exist")
		return 1

def edit_snappyHexMeshDict(dictionary, domain_json):
	try:
		tqdm.write("{}: Editing snappyHexMeshDict file: {}".format(datetime.datetime.now(),dictionary))
		snappyHexMeshDict = ParsedParameterFile(dictionary)

		with open(domain_json, 'r') as f:
			domain_dict = json.load(f)

		# clear old things
		# snappyHexMeshDict["geometry"]["domain_capped.stl"]["regions"] = ""

		# wall
		regions = {"vessel":{"name": "vessel"}}
		for key, value in domain_dict.items():
			try: 
				if value["type"] == "Stenosis":
					continue
				else:
					regions.update({key
						: {"name": key}})
			except:
				continue

		snappyHexMeshDict["geometry"]["domain_capped.stl"]["regions"] = regions

		# castellation control
		regions = {"vessel":{"level":"(3 4)", "patchInfo":{"type": "wall",}}}

		for key, value in domain_dict.items():
			try:
				if value["type"] == "inlet" or value["type"] == "outlet":
					regions.update({key: {"level": "(4 4)", "patchInfo":{"type":"patch"}}})
				else:
					continue
			except:
				continue

		snappyHexMeshDict["castellatedMeshControls"]["refinementSurfaces"]["Geometry"]["regions"] = regions

		# location in mesh
		snappyHexMeshDict["castellatedMeshControls"]["locationInMesh"] = "(" + \
			str(domain_dict["bifurcation_point"]["coordinate"][0]/1000) + " " + \
			str(domain_dict["bifurcation_point"]["coordinate"][1]/1000) + " " + \
			str(domain_dict["bifurcation_point"]["coordinate"][2]/1000) + ")"

		try:
			snappyHexMeshDict.writeFile()
		except IOError:
			tqdm.write("Can't write file. Content would have been:")
			tqdm.write(snappyHexMeshDict)

	except IOError:
		tqdm.write(snappyHexMeshDict, "does not exist")
		return 1

def stl_concat(domain_json):
	if not os.path.exists(domain_json):
		return

	working_dir = os.path.dirname(domain_json)

	with open(domain_json, 'r') as f:
		domain_dict = json.load(f)

	if os.path.exists(os.path.join(working_dir,"domain_capped.stl")):
		os.remove(os.path.join(working_dir,"domain_capped.stl"))

	# vessel wall
	# check nan and correct
	mesh = trimesh.load_mesh(os.path.join(working_dir,domain_dict["vessel"]["filename"]))
	mesh.process()
	mesh.export(os.path.join(working_dir,domain_dict["vessel"]["filename"]), file_type='stl_ascii')

	stl_text = open(os.path.join(working_dir,domain_dict["vessel"]["filename"])).read().splitlines(True)
	stl_text[0] = "solid vessel\n"
	stl_text.append("\n")

	fout = open(os.path.join(working_dir,"domain_capped.stl"), 'w')
	fout.writelines(stl_text)

	for key, value in domain_dict.items():
		try:
			if value["type"] == "Stenosis":
				continue
			else:
				# check nan and correct
				mesh = trimesh.load_mesh(os.path.join(working_dir,value["filename"]))
				mesh.process()
				mesh.export(os.path.join(working_dir,value["filename"]), file_type='stl_ascii') 

				stl_text = open(os.path.join(working_dir,value["filename"])).read().splitlines(True)
				stl_text[0] = "solid " + key + "\n"
				stl_text.append("\n")
				fout.writelines(stl_text)
		except:
			continue

	fout.close()

def run_case(case_dir, output_vtk=False, parallel=True, cores=4):
	startTime = datetime.datetime.now()

	tqdm.write("********************************* OpenFOAM CFD Operation *********************************")
	tqdm.write("{}: Execute OpenFOAM CFD simulation on directory: {}".format(datetime.datetime.now(),case_dir))

	tqdm.write("{}: STL domain merging...".format(datetime.datetime.now()))
	stl_concat(os.path.join(case_dir,"domain.json"))

	# copy surface from case directory
	tqdm.write("{}: Copying necessary files...".format(datetime.datetime.now()))
	source_file = os.path.join(case_dir,"domain_capped.stl")
	target_file = "./constant/triSurface/domain_capped.stl"
	shutil.copy(source_file, target_file)

	source_file = os.path.join(case_dir,"domain.json")
	target_file = "./constant/domain.json"
	shutil.copy(source_file, target_file)

	# clean workspace
	tqdm.write("{}: Cleaning workspace...".format(datetime.datetime.now()))
	if os.path.exists("./0/vorticity"):
		os.remove("./0/vorticity")
	if os.path.exists("./0/wallShearStress"):
		os.remove("./0/wallShearStress")
	if os.path.exists("./constant/polyMesh"):
		shutil.rmtree("./constant/polyMesh")
	if os.path.exists("./constant/extendedFeatureEdgeMesh"):
		shutil.rmtree("./constant/extendedFeatureEdgeMesh")
	for folder in os.listdir("./"):
		try:
			if folder == "0":
				continue
			is_cfd_result = float(folder)

			shutil.rmtree(os.path.join("./",folder))
		except ValueError:
			continue

	# blockMesh
	blockMeshDict_file = "./system/blockMeshDict"
	result = edit_blockMeshDict(blockMeshDict_file, "./constant/triSurface/domain_capped.stl")

	if result == 1:
		tqdm.write("blockMeshDict edit fail, case abort")
		return

	# create log dir
	os.makedirs("./log",exist_ok=True)

	tqdm.write("{}: Execute blockMesh...".format(datetime.datetime.now()))
	os.system("blockMesh > ./log/blockMesh.log")

	# extract surface features
	tqdm.write("{}: Execute surfaceFeatureExtract...".format(datetime.datetime.now()))
	os.system("surfaceFeatureExtract > ./log/surfaceFeatureExtract.log")

	if parallel:
		# remove all previously defined parallel outputs
		for folder in os.listdir("./"):
			if "processor" in folder:
				shutil.rmtree(folder)

		# decompose the mesh for multicore computation
		edit_decompseParDict("./system/decomposeParDict", cores=cores)
		tqdm.write("{}: Execute decompsePar...".format(datetime.datetime.now()))
		os.system("decomposePar > ./log/decomposePar.log")

		# need to edit snappyHexMesh file for meshing
		edit_snappyHexMeshDict("./system/snappyHexMeshDict", "./constant/domain.json")
		tqdm.write("{}: Execute snappyHexMesh in parallel...".format(datetime.datetime.now()))
		os.system("foamJob -parallel -screen snappyHexMesh -overwrite > ./log/snappyHexMesh.log")
		# os.system("foamJob -parallel snappyHexMesh -overwrite")

		# reconstruct mesh surface in parallel run
		tqdm.write("{}: Reconstructing mesh from parallel run...".format(datetime.datetime.now()))
		os.system("reconstructParMesh -latestTime -mergeTol 1E-06 -constant > ./log/reconstructParMesh.log")
		os.system("reconstructPar -latestTime > ./log/reconstructPar.log")

		# remove all previously defined parallel outputs
		for folder in os.listdir("./"):
			if "processor" in folder:
				shutil.rmtree(folder)

		# boundary conditions
		edit_velocity("./0/U", "./constant/domain.json", velocity=32.53)
		edit_pressure("./0/p", "./constant/domain.json", pressure=0)

		# decompose the mesh for multicore computation
		edit_decompseParDict("./system/decomposeParDict", cores=cores)
		tqdm.write("{}: Execute decompsePar...".format(datetime.datetime.now()))
		os.system("decomposePar > ./log/decomposePar.log")

		# CFD 
		# print("{}: Execute icoFoam in parallel...".format(datetime.datetime.now()))
		# os.system("foamJob -parallel -screen icoFoam > ./log/icoFoam.log")
		
		tqdm.write("{}: Execute pisoFoam in parallel...".format(datetime.datetime.now()))
		os.system("foamJob -parallel -screen pisoFoam > ./log/pisoFoam.log")

		# reconstruct mesh surface in parallel run
		tqdm.write("{}: Reconstructing mesh from parallel run...".format(datetime.datetime.now()))
		os.system("reconstructParMesh -mergeTol 1E-06 -constant > ./log/reconstructParMesh.log")
		os.system("reconstructPar > ./log/reconstructPar.log")

		# remove all previously defined parallel outputs
		for folder in os.listdir("./"):
			if "processor" in folder:
				shutil.rmtree(folder)

	else:
		# need to edit snappyHexMesh file for meshing
		edit_snappyHexMeshDict("./system/snappyHexMeshDict", "./constant/domain.json")
		tqdm.write("{}: Execute snappyHexMesh...".format(datetime.datetime.now()))
		os.system("snappyHexMesh -overwrite > ./log/snappyHexMesh.log")

		# run cfd
		# need to edit initial velocity file
		edit_velocity("./0/U", "./constant/domain.json", velocity=32.53)
		# print("{}: Execute icoFoam...".format(datetime.datetime.now()))
		# os.system("icoFoam > ./log/icoFoam.log")

		tqdm.write("{}: Execute pisoFoam...".format(datetime.datetime.now()))
		os.system("icoFoam > ./log/pisoFoam.log")

	# post processing
	tqdm.write("{}: Computing vorticity...".format(datetime.datetime.now()))
	os.system("pisoFoam -postProcess -func vorticity > ./log/vorticity.log")

	tqdm.write("{}: Computing wall shear stress...".format(datetime.datetime.now()))
	os.system("pisoFoam -postProcess -func wallShearStress > ./log/wallShearStress.log")

	# create OpenFOAM read fill for paraview
	os.system("touch OpenFOAM.OpenFOAM")

	# output as vtk file
	if output_vtk:
		tqdm.write("Convert to VTK output")
		os.system("foamToVTK > ./log/foamToVTK.log")

	endTime = datetime.datetime.now()
	tqdm.write("{}: Auto CFD pipeline complete, time elapsed: {}s".format(datetime.datetime.now(),(endTime-startTime).total_seconds()))

	# copy result back to storage node
	tqdm.write("{}: Copying result files...".format(datetime.datetime.now()))

	if os.path.exists(os.path.join(case_dir,"CFD_OpenFOAM")):
		shutil.rmtree(os.path.join(case_dir,"CFD_OpenFOAM"), ignore_errors=True)
	os.makedirs(os.path.join(case_dir,"CFD_OpenFOAM"))

	for folder in os.listdir("./"):
		try:
			is_cfd_result = float(folder)

			src_folder = os.path.join("./",folder)
			tgt_folder = os.path.join(case_dir,"CFD_OpenFOAM",folder)
			shutil.copytree(src_folder,tgt_folder)
		except ValueError:
			continue

	src_file = os.path.join("./","OpenFOAM.OpenFOAM")
	tgt_file = os.path.join(case_dir,"CFD_OpenFOAM","OpenFOAM.OpenFOAM")
	shutil.copy(src_file,tgt_file)

	src_folder = os.path.join("./","constant")
	tgt_folder = os.path.join(case_dir,"CFD_OpenFOAM","constant")
	shutil.copytree(src_folder,tgt_folder)

	src_folder = os.path.join("./","log")
	tgt_folder = os.path.join(case_dir,"CFD_OpenFOAM","log")
	shutil.copytree(src_folder,tgt_folder)

	if output_vtk:
		src_folder = os.path.join("./","VTK")
		tgt_folder = os.path.join(case_dir,"CFD_OpenFOAM","VTK")
		shutil.copytree(src_folder,tgt_folder)

	tqdm.write("{}: CFD operation on {} complete".format(datetime.datetime.now(),case_dir))

def main():
	data_dir = "/mnt/DIIR-JK-NAS/data/intracranial"
	sub_data_dirs = [
		"data_ESASIS_followup/medical",
		"data_ESASIS_followup/stent",
		"data_ESASIS_no_stenting",
		"data_surgery",
		"data_wingspan"
		]

	# data_dir = "/mnt/DIIR-JK-NAS/data/intracranial/data_30_30"
	# sub_data_dirs = ["surgery"]

	# phases = ["baseline", "baseline-post", "12months", "followup"]
	phases = ["baseline"]

	for sub_data_dir in sub_data_dirs:
		# datalist = os.listdir(os.path.join(data_dir,sub_data_dir))
		datalist = [
		 	"089",
		 	"133",
		 	]
		
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

				run_case(os.path.join(data_dir,sub_data_dir,case,phase),output_vtk=True, parallel=True, cores=8)

			# run_case(os.path.join(data_dir,sub_data_dir,case),output_vtk=True, parallel=True, cores=8)

if __name__ == "__main__":
	main()