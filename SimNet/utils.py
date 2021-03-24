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
	reader = vtk