import os
import shutil
from PyFoam.RunDictionary.ParsedBlockMeshDict import ParsedBlockMeshDict
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
import vtk
import datetime
import json
from tqdm import tqdm
import trimesh
import tempfile

def edit_blockMeshDict(dictionary, stl, cellNumber=40, edge_buffer=2):
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
		tqdm.write(blockMeshDict_file, " does not exist")
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
		tqdm.write(dictionary, " does not exist")
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
		tqdm.write(dictionary, " does not exist")
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
		tqdm.write(dictionary, " does not exist")
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
				if value["type"] in ["Stenosis","DoS_Ref","Bifurcation","Others"]:
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
				if value["type"] in ["inlet","outlet"]:
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
		tqdm.write(dictionary + " does not exist")
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
			if value["type"] in ["Stenosis","DoS_Ref","Bifurcation","Others"]:
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

def run_case(case_dir, output_vtk=False, parallel=True, cores=4, cellNumber=40, output_dir_name="CFD_OpenFOAM"):
	startTime = datetime.datetime.now()

	tqdm.write("********************************* OpenFOAM CFD Operation *********************************")
	tqdm.write("{}: Execute OpenFOAM CFD simulation on directory: {}".format(datetime.datetime.now(),case_dir))

	tqdm.write("{}: STL domain merging...".format(datetime.datetime.now()))
	stl_concat(os.path.join(case_dir,"domain.json"))

	# create temporary working directory
	tmp_dir = tempfile.TemporaryDirectory(prefix="tmp_",dir="./")

	# copy surface from case directory
	tqdm.write("{}: Copying necessary files...".format(datetime.datetime.now()))
	source_file = os.path.join(case_dir,"domain_capped.stl")
	surface_file = os.path.join(tmp_dir.name,"constant","triSurface","domain_capped.stl")
	if not os.path.exists(os.path.dirname(surface_file)):
		os.makedirs(os.path.dirname(surface_file))
	shutil.copy(source_file, surface_file)

	source_file = os.path.join(case_dir,"domain.json")
	domain_file = os.path.join(tmp_dir.name,"constant","domain.json")
	if not os.path.exists(os.path.dirname(domain_file)):
		os.makedirs(os.path.dirname(domain_file))
	shutil.copy(source_file, domain_file)

	# copy necessary dict to tmp dir
	source_files = [
		"./constant/transportProperties",
		"./constant/turbulenceProperties",
		"./system/blockMeshDict",
		"./system/controlDict",
		"./system/surfaceFeaturesDict",
		"./system/decomposeParDict",
		"./system/snappyHexMeshDict",
		"./system/fvSchemes",
		"./system/fvSolution",
		"./system/meshQualityDict",
		"./0/U",
		"./0/p",
	]

	for source_file in source_files:
		target_file = os.path.join(tmp_dir.name,source_file)
		if not os.path.exists(os.path.dirname(target_file)):
			os.makedirs(os.path.dirname(target_file))
		shutil.copy(source_file, target_file)

	# blockMesh
	blockMeshDict_file = os.path.join(tmp_dir.name,"system","blockMeshDict")
	result = edit_blockMeshDict(blockMeshDict_file, surface_file,cellNumber=cellNumber)

	if result == 1:
		tqdm.write("blockMeshDict edit fail, case abort")
		return

	# create log dir
	os.makedirs(os.path.join(tmp_dir.name,"log"),exist_ok=True)

	tqdm.write("{}: Execute blockMesh...".format(datetime.datetime.now()))
	os.system("blockMesh -case {} > {}".format(
		tmp_dir.name,
		os.path.join(tmp_dir.name,"log","blockMesh.log")))

	# # extract surface features
	# tqdm.write("{}: Execute surfaceFeatures...".format(datetime.datetime.now()))
	# os.system("surfaceFeatures -case {} > {}".format(
	# 	tmp_dir.name,
	# 	os.path.join(tmp_dir.name,"log","surfaceFeatures.log")))

	# if parallel:
	# 	# decompose the mesh for multicore computation
	# 	edit_decompseParDict(os.path.join(tmp_dir.name,"system","decomposeParDict"), cores=cores)
	# 	tqdm.write("{}: Execute decompsePar...".format(datetime.datetime.now()))
	# 	os.system("decomposePar -case {} > {}".format(
	# 		tmp_dir.name,
	# 		os.path.join(tmp_dir.name,"log","decomposePar_1.log")))

	# 	# need to edit snappyHexMesh file for meshing
	# 	edit_snappyHexMeshDict(os.path.join(tmp_dir.name,"system","snappyHexMeshDict"), os.path.join(tmp_dir.name,"constant","domain.json"))
	# 	tqdm.write("{}: Execute snappyHexMesh in parallel...".format(datetime.datetime.now()))
	# 	os.system("foamJob -parallel -screen -case {} snappyHexMesh -overwrite > {}".format(
	# 		tmp_dir.name,
	# 		os.path.join(tmp_dir.name,"log","snappyHexMesh.log")
	# 		))

	# 	# reconstruct mesh surface in parallel run
	# 	tqdm.write("{}: Reconstructing mesh from parallel run...".format(datetime.datetime.now()))
	# 	os.system("reconstructParMesh -latestTime -mergeTol 1E-06 -constant -case {} > {}".format(
	# 		tmp_dir.name,
	# 		os.path.join(tmp_dir.name,"log","reconstructParMesh_2.log")
	# 		))
	# 	os.system("reconstructPar -latestTime -case {} > {}".format(
	# 		tmp_dir.name,
	# 		os.path.join(tmp_dir.name,"log","reconstructPar_1.log")
	# 		))

	# 	# remove all previously defined parallel outputs
	# 	for folder in os.listdir(tmp_dir.name):
	# 		if "processor" in folder:
	# 			shutil.rmtree(os.path.join(tmp_dir.name,folder))

	# 	# boundary conditions
	# 	edit_velocity(os.path.join(tmp_dir.name,"0","U"), os.path.join(tmp_dir.name,"constant","domain.json"), velocity=32.53)
	# 	edit_pressure(os.path.join(tmp_dir.name,"0","p"), os.path.join(tmp_dir.name,"constant","domain.json"), pressure=0)

	# 	# decompose the mesh for multicore computation
	# 	edit_decompseParDict(os.path.join(tmp_dir.name,"system","decomposeParDict"), cores=cores)
	# 	tqdm.write("{}: Execute decompsePar...".format(datetime.datetime.now()))
	# 	os.system("decomposePar -case {} > {}".format(
	# 		tmp_dir.name,
	# 		os.path.join(tmp_dir.name,"log","decomposePar_2.log")
	# 		))

	# 	# CFD 
	# 	# print("{}: Execute icoFoam in parallel...".format(datetime.datetime.now()))
	# 	# os.system("foamJob -parallel -screen icoFoam > ./log/icoFoam.log")
		
	# 	tqdm.write("{}: Execute pisoFoam in parallel...".format(datetime.datetime.now()))
	# 	os.system("foamJob -parallel -screen -case {} pisoFoam > {}".format(
	# 		tmp_dir.name,
	# 		os.path.join(tmp_dir.name,"log","pisoFoam.log")
	# 		))

	# 	# reconstruct mesh surface in parallel run
	# 	tqdm.write("{}: Reconstructing mesh from parallel run...".format(datetime.datetime.now()))
	# 	os.system("reconstructParMesh -mergeTol 1E-06 -constant -case {} > {}".format(
	# 		tmp_dir.name,
	# 		os.path.join(tmp_dir.name,"log","reconstructParMesh_2.log")
	# 		))
	# 	os.system("reconstructPar -case {} > {}".format(
	# 		tmp_dir.name,
	# 		os.path.join(tmp_dir.name,"log","reconstructPar_2.log")
	# 		))
	# else:
	# 	# need to edit snappyHexMesh file for meshing
	# 	edit_snappyHexMeshDict(os.path.join(tmp_dir.name,"system","snappyHexMeshDict"), os.path.join(tmp_dir.name,"constant","domain.json"))
	# 	tqdm.write("{}: Execute snappyHexMesh...".format(datetime.datetime.now()))
	# 	os.system("snappyHexMesh -overwrite -case {} > {}".format(
	# 		tmp_dir.name,
	# 		os.path.join(tmp_dir.name,"log","snappyHexMesh.log")
	# 		))

	# 	# run cfd
	# 	# need to edit initial velocity and pressure file
	# 	edit_velocity(os.path.join(tmp_dir.name,"0","U"), os.path.join(tmp_dir.name,"constant","domain.json"), velocity=32.53)
	# 	edit_pressure(os.path.join(tmp_dir.name,"0","p"), os.path.join(tmp_dir.name,"constant","domain.json"), pressure=0)

	# 	# print("{}: Execute icoFoam...".format(datetime.datetime.now()))
	# 	# os.system("icoFoam > ./log/icoFoam.log")

	# 	tqdm.write("{}: Execute pisoFoam...".format(datetime.datetime.now()))
	# 	os.system("icoFoam -case {} > {}".format(
	# 		tmp_dir.name,
	# 		os.path.join(tmp_dir.name,"log","pisoFoam.log")
	# 		))

	# # post processing
	# tqdm.write("{}: Computing vorticity...".format(datetime.datetime.now()))
	# os.system("pisoFoam -postProcess -func vorticity -case {} > {}".format(
	# 	tmp_dir.name,
	# 	os.path.join(tmp_dir.name,"log","vorticity.log")
	# 	))

	# tqdm.write("{}: Computing wall shear stress...".format(datetime.datetime.now()))
	# os.system("pisoFoam -postProcess -func wallShearStress -case {} > {}".format(
	# 	tmp_dir.name,
	# 	os.path.join(tmp_dir.name,"log","wallShearStress.log")
	# 	))

	# create OpenFOAM read fill for paraview
	os.system("touch ./{}/OpenFOAM.OpenFOAM".format(tmp_dir.name))

	# output as vtk file
	if output_vtk:
		tqdm.write("Convert to VTK output")
		os.system("foamToVTK -case {}> {}".format(
			tmp_dir.name,
			os.path.join(tmp_dir.name,"log","foamToVTK.log")
		))

		for file in os.listdir(os.path.join(tmp_dir.name,"VTK")):
			src = os.path.join(tmp_dir.name,"VTK",file)
			if not os.path.isfile(os.path.join(src)):
				continue

			if not len(file.split("_")) > 2:
				continue

			output_filename = "OpenFOAM_{}.vtk".format(file.split(".")[0].split("_")[-1])

			tgt = os.path.join(tmp_dir.name,"VTK",output_filename)

			if file.split(".")[1] == "vtk" and file.split("_")[0]!="OpenFOAM":
				 tqdm.write("Rename: {} to {}".format(src,tgt))
				 os.rename(src,tgt)


	endTime = datetime.datetime.now()
	tqdm.write("{}: Auto CFD pipeline complete, time elapsed: {:.2f}s".format(datetime.datetime.now(),(endTime-startTime).total_seconds()))

	# copy result back to storage node
	tqdm.write("{}: Copying result files...".format(datetime.datetime.now()))

	if os.path.exists(os.path.join(case_dir,output_dir_name)):
		shutil.rmtree(os.path.join(case_dir,output_dir_name), ignore_errors=True)
	os.makedirs(os.path.join(case_dir,output_dir_name),exist_ok=True)

	for folder in os.listdir(tmp_dir.name):
		try:
			is_cfd_result = float(folder)

			src_folder = os.path.join(tmp_dir.name,folder)
			tgt_folder = os.path.join(case_dir,output_dir_name,folder)
			shutil.copytree(src_folder,tgt_folder)
		except ValueError:
			continue

	src_file = os.path.join(tmp_dir.name,"OpenFOAM.OpenFOAM")
	tgt_file = os.path.join(case_dir,output_dir_name,"OpenFOAM.OpenFOAM")
	shutil.copy(src_file,tgt_file)

	src_folder = os.path.join(tmp_dir.name,"constant")
	tgt_folder = os.path.join(case_dir,output_dir_name,"constant")
	shutil.copytree(src_folder,tgt_folder)

	src_folder = os.path.join(tmp_dir.name,"log")
	tgt_folder = os.path.join(case_dir,output_dir_name,"log")
	shutil.copytree(src_folder,tgt_folder)

	if output_vtk:
		src_folder = os.path.join(tmp_dir.name,"VTK")
		tgt_folder = os.path.join(case_dir,output_dir_name,"VTK")
		shutil.copytree(src_folder,tgt_folder)

	tqdm.write("{}: CFD operation on {} complete".format(datetime.datetime.now(),case_dir))