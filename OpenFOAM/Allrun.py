import os
import shutil
from PyFoam.RunDictionary.ParsedBlockMeshDict import ParsedBlockMeshDict
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
import vtk
import datetime
import json

def edit_blockMeshDict(dictionary, stl, edge_buffer=2):
	try:
		print("{}: Editing blockMeshDict: {}".format(datetime.datetime.now(),dictionary))
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

		try:
			blockMeshDict.writeFile()
		except IOError:
			print("Can't write file. Content would have been:")
			print(blockMeshDict)

		return 0

	except IOError:
		print(blockMeshDict_file, "does not exist")
		return 1

def edit_decompseParDict(dictionary, cores=4):
	try:
		print("{}: Editing decompseParDict: {}".format(datetime.datetime.now(),dictionary))
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
			print("Can't write file. Content would have been:")
			print(decompseParDict)

	except IOError:
		print(dictionary, "does not exist")
		return 1	

def edit_velocity(dictionary, inlet_json, velocity=1):
	try:
		print("{}: Editing velocity file: {}".format(datetime.datetime.now(),dictionary))
		velocityDict = ParsedParameterFile(dictionary)

		with open(inlet_json, 'r') as f:
			inlet_dict = json.load(f)

		velocityDict["boundaryField"]["ICA"]["value"] = "uniform (" + \
			str(inlet_dict["ICA"]["tangent"][0]/1000*velocity) + " " + \
			str(inlet_dict["ICA"]["tangent"][1]/1000*velocity) + " " + \
			str(inlet_dict["ICA"]["tangent"][2]/1000*velocity) + ")"

		try:
			velocityDict.writeFile()
		except IOError:
			print("Can't write file. Content would have been:")
			print(velocityDict)

	except IOError:
		print(dictionary, "does not exist")
		return 1

def edit_snappyHexMeshDict(dictionary, inlet_json):
	try:
		print("{}: Editing snappyHexMeshDict file: {}".format(datetime.datetime.now(),dictionary))
		snappyHexMeshDict = ParsedParameterFile(dictionary)

		with open(inlet_json, 'r') as f:
			inlet_dict = json.load(f)

		snappyHexMeshDict["castellatedMeshControls"]["locationInMesh"] = "(" + \
			str(inlet_dict["BifurcationPoint"]["coordinate"][0]/1000) + " " + \
			str(inlet_dict["BifurcationPoint"]["coordinate"][1]/1000) + " " + \
			str(inlet_dict["BifurcationPoint"]["coordinate"][2]/1000) + ")"

		try:
			snappyHexMeshDict.writeFile()
		except IOError:
			print("Can't write file. Content would have been:")
			print(snappyHexMeshDict)

	except IOError:
		print(dictionary, "does not exist")
		return 1

def run_case(case_dir, output_vtk=False, parallel=True, cores=4):
	startTime = datetime.datetime.now()

	print("{}: Execute OpenFOAM CFD simulation on directory: {}".format(datetime.datetime.now(),case_dir))

	# copy surface from case directory
	print("{}: Copying necessary files...".format(datetime.datetime.now()))
	source_file = os.path.join(case_dir,"surface_capped.stl")
	target_file = "./constant/triSurface/surface_capped.stl"
	shutil.copy(source_file, target_file)

	source_file = os.path.join(case_dir,"inlets.json")
	target_file = "./constant/inlets.json"
	shutil.copy(source_file, target_file)

	# clean workspace
	print("{}: Cleaning workspace...".format(datetime.datetime.now()))
	if os.path.exists("./constant/polyMesh"):
		shutil.rmtree("./constant/polyMesh")
	if os.path.exists("./constant/extendedFeatureEdgeMesh"):
		shutil.rmtree("./constant/extendedFeatureEdgeMesh")

	# blockMesh
	blockMeshDict_file = "./system/blockMeshDict"
	result = edit_blockMeshDict(blockMeshDict_file, "./constant/triSurface/surface_capped.stl")
	if result == 1:
		print("blockMeshDict edit fail, case abort")
		return

	# create log dir
	os.makedirs("./log",exist_ok=True)

	print("{}: Execute blockMesh...".format(datetime.datetime.now()))
	os.system("blockMesh > ./log/blockMesh.log")

	# extract surface features
	print("{}: Execute surfaceFeatureExtract...".format(datetime.datetime.now()))
	os.system("surfaceFeatureExtract > ./log/surfaceFeatureExtract.log")

	if parallel:
		# remove all previously defined parallel outputs
		for folder in os.listdir("./"):
			if "processor" in folder:
				shutil.rmtree(folder)

		# decompose the mesh for multicore computation
		edit_decompseParDict("./system/decomposeParDict", cores=cores)
		print("{}: Execute decompsePar...".format(datetime.datetime.now()))
		os.system("decomposePar > ./log/decomposePar.log")

		# need to edit snappyHexMesh file for meshing
		edit_snappyHexMeshDict("./system/snappyHexMeshDict", "./constant/inlets.json")
		print("{}: Execute snappyHexMesh in parallel...".format(datetime.datetime.now()))
		os.system("foamJob -parallel -screen snappyHexMesh -overwrite > ./log/snappyHexMesh.log")
		# os.system("foamJob -parallel snappyHexMesh -overwrite")

		# reconstruct mesh surface in parallel run
		print("{}: Reconstructing mesh from parallel run...".format(datetime.datetime.now()))
		os.system("reconstructParMesh -latestTime -mergeTol 1E-06 -constant > ./log/reconstructParMesh.log")

		os.system("reconstructPar -latestTime > ./log/reconstructPar.log")

		# remove all previously defined parallel outputs
		for folder in os.listdir("./"):
			if "processor" in folder:
				shutil.rmtree(folder)

		# decompose the mesh for multicore computation
		edit_decompseParDict("./system/decomposeParDict", cores=cores)
		print("{}: Execute decompsePar...".format(datetime.datetime.now()))
		os.system("decomposePar > ./log/decomposePar.log")

		# CFD 
		edit_velocity("./0/U", "./constant/inlets.json", velocity=1.8)
		print("{}: Execute icoFoam in parallel...".format(datetime.datetime.now()))
		os.system("foamJob -parallel -screen icoFoam > ./log/icoFoam.log")

		# reconstruct mesh surface in parallel run
		print("{}: Reconstructing mesh from parallel run...".format(datetime.datetime.now()))
		os.system("reconstructParMesh -mergeTol 1E-06 -constant > ./log/reconstructParMesh.log")

		os.system("reconstructPar > ./log/reconstructPar.log")

		# remove all previously defined parallel outputs
		for folder in os.listdir("./"):
			if "processor" in folder:
				shutil.rmtree(folder)

	else:
		# need to edit snappyHexMesh file for meshing
		edit_snappyHexMeshDict("./system/snappyHexMeshDict", "./constant/inlets.json")
		print("{}: Execute snappyHexMesh...".format(datetime.datetime.now()))
		os.system("snappyHexMesh -overwrite > ./log/snappyHexMesh.log")

		# run cfd
		# need to edit initial velocity file
		edit_velocity("./0/U", "./constant/inlets.json", velocity=1.8)
		print("{}: Execute icoFoam...".format(datetime.datetime.now()))
		os.system("icoFoam > ./log/icoFoam.log")

	# create OpenFOAM read fill for paraview
	os.system("touch OpenFOAM.OpenFOAM")

	# output as vtk file
	if output_vtk:
		print("Convert to VTK output")
		os.system("foamToVTK > ./log/foamToVTK.log")

	endTime = datetime.datetime.now()
	print("{}: Auto CFD pipeline complete, time elapsed: {}s".format(datetime.datetime.now(),(endTime-startTime).total_seconds()))

def main():
	data_dir = "/mnt/DIIR-JK-NAS/data/intracranial/followup/medical"

	phases = ["baseline", "baseline-post", "12months", "followup"]

	# for case in os.listdir(data_dir):
	for case in ["ChanSP"]:
		for phase in phases:
			run_case(os.path.join(data_dir,case,phase),output_vtk=False, parallel=True, cores=4)
			exit()

if __name__ == "__main__":
	main()