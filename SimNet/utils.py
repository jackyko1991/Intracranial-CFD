import os
import trimesh
import vtk
import json

def stl_concat(domain_json):
	print("stl_concat")

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

def OpenFOAM_result_to_csv(case_list,output_path):
	# average filter
	averageFilter = vtk.vtkTemporalStatistics()

	for case in case_list:
		reader = vtk.vtkUnstructuredGridReader()
		reader.SetFileName(case)
		reader.Update() 

		# scale up the CFD result
		transform = vtk.vtkTransform()
		transform.Scale(1000,1000,1000)

		transformFilter = vtk.vtkTransformFilter()
		transformFilter.SetInputData(reader.GetOutput())
		transformFilter.SetTransform(transform)
		transformFilter.Update()

		averageFilter.SetInputData(transformFilter.GetOutput())
		averageFilter.Update()

	# work around for polydata to unstructured grid
	appendFilter = vtk.vtkAppendFilter()
	appendFilter.AddInputData(averageFilter.GetOutput())
	appendFilter.Update();

	unstructuredGrid = vtk.vtkUnstructuredGrid();
	unstructuredGrid.ShallowCopy(appendFilter.GetOutput())

	# print("transform filter: ",transformFilter.GetOutput().GetNumberOfCells())
	# os.makedirs(os.path.basename(output_path),exist_ok=True)

	# writer = vtk.vtkXMLUnstructuredGridWriter()
	# writer.SetFileName(output_path)
	# writer.SetInputData(unstructuredGrid)
	# writer.Update()

	table = vtk.vtkDataObjectToTable()
	table.SetInputData(averageFilter.GetOutput())
	table.Update()
	table.GetOutput().AddColumn(averageFilter.GetOutput().GetPoints().GetData())
	table.Update()

	writer = vtk.vtkDelimitedTextWriter()
	writer.SetInputConnection(table.GetOutputPort())
	writer.SetFileName(output_path)
	writer.Update()
	writer.Write()