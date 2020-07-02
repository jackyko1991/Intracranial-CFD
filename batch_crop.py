import os
import shutil
import vtk
import math
import numpy as np
import json

# data_dir = "Z:/data/intracranial/followup/medical"
data_dir = "/mnt/DIIR-JK-NAS/data/intracranial/followup/medical"
binary_path = "D:/projects/CFD_intracranial/cxx/Vessel-Centerline-Extraction/build/Release/CenterlineExtraction.exe"
dist_from_bif_inlet = 35
dist_from_bif_outlet = 25
smooth_relaxation_3DRA = 0.1
smooth_relaxation_CBCT = 0.2
batch_centerline = False
batch_clip = True
capping = False
stl_concat = True

def clip_polydata_by_box(polydata, point, tangent, normal, binormal, size=[10,10,1], capping=True):
	"""Clip the input polydata with given box shape and direction

	Parameters:
	polydata (polydata): Input polydata
	point (array): Coordinate of clipping box center
	normal (array): Direction tangent of the clipping box (major axis)
	normal (array): Direction normal of the clipping box
	normal (array): Direction binormal of the clipping box
	size (array): Size of the clipping box (default: [10,10,1])
	capping (bool): Flag to cap the clipped surface

	Returns:
	dict: {"clipped_surface": clipped polydata, "clip_box": clipping box, "clip_plane": clipping plane, "boundary_cap": cap}
	"""

	# create a clipping box widget
	clipWidget = vtk.vtkBoxWidget()
	transform = vtk.vtkTransform()

	transform.Translate(point)	
	w = math.atan(math.sqrt(tangent[0]**2+tangent[1]**2)/tangent[2])*180/3.14
	transform.RotateWXYZ(w, -tangent[1], tangent[0],0)
	transform.Scale(size)

	clipBox = vtk.vtkCubeSource()
	transformFilter = vtk.vtkTransformPolyDataFilter()
	transformFilter.SetInputConnection(clipBox.GetOutputPort())
	transformFilter.SetTransform(transform)
	transformFilter.Update()

	clipWidget.SetTransform(transform)
	clipFunction = vtk.vtkPlanes()
	clipWidget.GetPlanes(clipFunction)

	clipper = vtk.vtkClipPolyData()
	clipper.SetClipFunction(clipFunction)
	clipper.SetInputData(polydata)
	clipper.GenerateClippedOutputOn()
	clipper.SetValue(0.0)
	clipper.Update()

	polydata.DeepCopy(clipper.GetOutput())

	# extract feature edges
	boundaryEdges = vtk.vtkFeatureEdges()
	boundaryEdges.SetInputData(polydata)
	boundaryEdges.BoundaryEdgesOn()
	boundaryEdges.FeatureEdgesOff()
	boundaryEdges.NonManifoldEdgesOff()
	boundaryEdges.ManifoldEdgesOff()
	boundaryEdges.Update()

	boundaryStrips = vtk.vtkStripper()
	boundaryStrips.SetInputData(boundaryEdges.GetOutput())
	boundaryStrips.Update()

	# Change the polylines into polygons
	boundaryPoly = vtk.vtkPolyData()
	boundaryPoly.SetPoints(boundaryStrips.GetOutput().GetPoints())
	boundaryPoly.SetPolys(boundaryStrips.GetOutput().GetLines())

	if capping:
		appendFilter = vtk.vtkAppendPolyData()
		appendFilter.AddInputData(polydata)
		appendFilter.AddInputData(boundaryPoly)
		appendFilter.Update()

		cleaner = vtk.vtkCleanPolyData()
		cleaner.SetInputData(appendFilter.GetOutput())
		cleaner.Update()
		polydata.DeepCopy(cleaner.GetOutput())

	# clipping plane
	point1 = []
	point2 = []
	origin = []
	
	for i in range(3):
		point1.append(point[i]-normal[i]*math.sqrt(2)/2*size[0]/2)
		point2.append(point[i]+normal[i]*math.sqrt(2)/2*size[1]/2)
		origin.append(point[i]+binormal[i]*math.sqrt(2)/2*size[0]/2)

	transform_clipbox = vtk.vtkTransform()
	transform_clipbox.Translate(point)	
	w = math.atan(math.sqrt(tangent[0]**2+tangent[1]**2)/tangent[2])*180/3.14
	transform_clipbox.RotateWXYZ(w, -tangent[1], tangent[0],0)
	transform_clipbox.Scale([number/2 for number in size])

	clipBox = vtk.vtkCubeSource()
	transformFilterClipBox = vtk.vtkTransformPolyDataFilter()
	transformFilterClipBox.SetInputConnection(clipBox.GetOutputPort())
	transformFilterClipBox.SetTransform(transform_clipbox)
	transformFilterClipBox.Update()

	clipBoxPolyData = vtk.vtkPolyData()
	clipBoxPolyData.DeepCopy(transformFilterClipBox.GetOutput())

	planeSource = vtk.vtkPlaneSource()
	planeSource.SetResolution(10,10)
	planeSource.SetOrigin(origin)
	planeSource.SetPoint1(point1)
	planeSource.SetPoint2(point2)
	planeSource.Update()

	clipPlanePolyData = vtk.vtkPolyData()
	clipPlanePolyData.DeepCopy(planeSource.GetOutput())
	return {"clipped_surface": polydata, "clip_box": clipBoxPolyData, "clip_plane": clipPlanePolyData, "boundary_cap": boundaryPoly}

def centerlineCalculation(working_dir, relaxation = 0):
	if not os.path.exists(os.path.join(working_dir,"surface.vtk")):
		return

	# convert vtk to stl
	reader = vtk.vtkGenericDataObjectReader()
	reader.SetFileName(os.path.join(working_dir,"surface.vtk"))
	reader.Update()
	surface = reader.GetOutput()
	
	if relaxation > 0:
		smoothFilter = vtk.vtkSmoothPolyDataFilter()
		smoothFilter.SetInputData(surface)
		smoothFilter.SetNumberOfIterations(100)
		smoothFilter.SetRelaxationFactor(relaxation)
		smoothFilter.FeatureEdgeSmoothingOff()
		smoothFilter.BoundarySmoothingOn()
		smoothFilter.Update()
		surface.DeepCopy(smoothFilter.GetOutput())

	writer = vtk.vtkSTLWriter()
	writer.SetFileName(os.path.join(working_dir,"surface.stl"))
	writer.SetInputData(surface)
	writer.Update()

	# compute centerline
	command = binary_path + " " + \
		os.path.join(working_dir,"surface.stl") + " " + \
		os.path.join(working_dir,"surface_capped.stl") + " " + \
		os.path.join(working_dir,"centerline.vtp")
	os.system(command)

def normalizeVessels(case_dir):
	phases = ["baseline","baseline-post","12months","followup"]

	centerlines = {}
	surfaces = {}

	for phase in phases:
		if not os.path.exists(os.path.join(case_dir,phase,"centerline.vtp")):
			continue
		# load the centerline file
		reader = vtk.vtkXMLPolyDataReader()
		reader.SetFileName(os.path.join(case_dir,phase,"centerline.vtp"))
		reader.Update()
		centerline = reader.GetOutput()
		centerlines.update({phase: centerline})

		# load surface files
		if not os.path.exists(os.path.join(case_dir,phase,"surface.stl")):
			continue
		# reader = vtk.vtkGenericDataObjectReader()
		reader = vtk.vtkSTLReader()
		reader.SetFileName(os.path.join(case_dir,phase,"surface.stl"))
		reader.Update()
		surface = reader.GetOutput()
		surfaces.update({phase: surface})

	# get the bifurcation point from baseline data
	# split the polydata by centerline id
	splitter = vtk.vtkThreshold()
	splitter.SetInputData(centerlines["baseline"])
	splitter.ThresholdBetween(0,0)
	splitter.SetInputArrayToProcess(0,0,0,vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS,"GroupIds")
	splitter.Update()
	mainBranch = splitter.GetOutput()

	maxAbsc = 0
	maxAbscId = 0

	for i in range(mainBranch.GetNumberOfPoints()):
		absc = mainBranch.GetPointData().GetArray("Abscissas").GetComponent(i,0)
		if absc > maxAbsc:
			maxAbsc = absc
			maxAbscId = i

	bifPoint = mainBranch.GetPoint(maxAbscId)

	bifPoint_list = {}
	bifPoint_absc_list = {}
	bifPoint_id_list = {}
	endPoint1_absc_list = {}
	endPoint1_id_list = {}
	endPoint2_absc_list = {}
	endPoint2_id_list = {}

	for key, centerline in centerlines.items():
		kdTree = vtk.vtkKdTreePointLocator()
		kdTree.SetDataSet(centerline)
		iD = kdTree.FindClosestPoint(bifPoint)

		bifPoint_absc_list.update({key: centerline.GetPointData().GetArray("Abscissas").GetComponent(iD,0)})
		bifPoint_id_list.update({key: iD})

		splitter = vtk.vtkThreshold()
		splitter.SetInputData(centerline)
		splitter.SetInputArrayToProcess(0,0,0,vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS,"GroupIds")
		splitter.ThresholdBetween(2,2)
		splitter.Update()
		ACA = splitter.GetOutput()
		endPoint1_absc_list.update({key: ACA.GetPointData().GetArray("Abscissas").GetComponent(ACA.GetNumberOfPoints()-1,0) - bifPoint_absc_list[key]})
		endPoint1_id_list.update({key: ACA.GetNumberOfPoints()-1})

		splitter.ThresholdBetween(3,3)
		splitter.Update()
		MCA = splitter.GetOutput()
		endPoint2_absc_list.update({key: ACA.GetPointData().GetArray("Abscissas").GetComponent(MCA.GetNumberOfPoints()-1,0) - bifPoint_absc_list[key]})
		endPoint2_id_list.update({key: ACA.GetNumberOfPoints()-1})

		# append the bifurcation point
		bifPoint_list.update({key: bifPoint})

	# get the start point coordinate
	start_id = 0
	start_point_absc = 0
	start_point = [0,0,0]
	start_point_tangent = [1,0,0]
	start_point_normal = [0,1,0]
	start_point_binormal = [0,0,1]

	for i in range(centerlines["baseline"].GetNumberOfPoints()):
		if (bifPoint_absc_list["baseline"] - centerlines["baseline"].GetPointData().GetArray("Abscissas").GetComponent(i,0) < min(bifPoint_absc_list.values())) and \
			(bifPoint_absc_list["baseline"] - centerlines["baseline"].GetPointData().GetArray("Abscissas").GetComponent(i,0) < dist_from_bif_inlet):
			break
		else:
			start_id = i
			start_point_absc = centerlines["baseline"].GetPointData().GetArray("Abscissas").GetComponent(i,0)
			start_point = list(centerlines["baseline"].GetPoint(i))
			start_point_tangent = list(centerlines["baseline"].GetPointData().GetArray("FrenetTangent").GetTuple(i))
			start_point_normal = list(centerlines["baseline"].GetPointData().GetArray("FrenetNormal").GetTuple(i))
			start_point_binormal = list(centerlines["baseline"].GetPointData().GetArray("FrenetBinormal").GetTuple(i))

	print("start_point:",start_point)

	# get the end point coordinates
	end_ids = []
	end_points = []
	end_points_tangent = []
	end_points_normal = []
	end_points_binormal = []
	splitter = vtk.vtkThreshold()
	splitter.SetInputData(centerlines["baseline"])
	splitter.SetInputArrayToProcess(0,0,0,vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS,"GroupIds")

	groupdIds = [2,3]

	for groupId in groupdIds:
		splitter.ThresholdBetween(groupId,groupId)
		splitter.Update()
		splitted_centerline = splitter.GetOutput()

		end_id = splitted_centerline.GetNumberOfPoints()-1
		end_point_absc = splitted_centerline.GetPointData().GetArray("Abscissas").GetComponent(splitted_centerline.GetNumberOfPoints()-1,0)
		end_point = list(splitted_centerline.GetPoints().GetPoint(splitted_centerline.GetNumberOfPoints()-1))
		end_point_tangent = list(splitted_centerline.GetPointData().GetArray("FrenetTangent").GetTuple(splitted_centerline.GetNumberOfPoints()-1))
		end_point_normal = list(splitted_centerline.GetPointData().GetArray("FrenetNormal").GetTuple(splitted_centerline.GetNumberOfPoints()-1))
		end_point_binormal = list(splitted_centerline.GetPointData().GetArray("FrenetBinormal").GetTuple(splitted_centerline.GetNumberOfPoints()-1))

		for i in range(splitted_centerline.GetNumberOfPoints()):
			if groupId == 2:
				endPoint_absc_list = endPoint1_absc_list.values()
			else:
				endPoint_absc_list = endPoint2_absc_list.values()

			if (splitter.GetOutput().GetPointData().GetArray("Abscissas").GetComponent(i,0)-bifPoint_absc_list["baseline"]< min(endPoint_absc_list)) and \
				(splitter.GetOutput().GetPointData().GetArray("Abscissas").GetComponent(i,0)-bifPoint_absc_list["baseline"] < dist_from_bif_outlet):
				end_id = i
				end_point_absc = splitted_centerline.GetPointData().GetArray("Abscissas").GetComponent(i,0)
				end_point = list(splitted_centerline.GetPoints().GetPoint(i))
				end_point_tangent = list(splitted_centerline.GetPointData().GetArray("FrenetTangent").GetTuple(i))
				end_point_normal = list(splitted_centerline.GetPointData().GetArray("FrenetNormal").GetTuple(i))
				end_point_binormal = list(splitted_centerline.GetPointData().GetArray("FrenetBinormal").GetTuple(i))

			else:
				end_ids.append(end_id)
				end_points.append(end_point)
				end_points_tangent.append(end_point_tangent)
				end_points_normal.append(end_point_normal)
				end_points_binormal.append(end_point_binormal)
				break

			if i == splitted_centerline.GetNumberOfPoints()-1:
				end_ids.append(end_id)
				end_points.append(end_point)
				end_points_tangent.append(end_point_tangent)
				end_points_normal.append(end_point_normal)
				end_points_binormal.append(end_point_binormal)

	# clip the surfaces
	clipBoxes = {}
	clipPlanes = {}
	boundaryCaps = {}
	surfaces_clipped = {}
	centerlines_clipped = {}
	# connected component calculation on surface and centerline
	connectedFilter = vtk.vtkConnectivityFilter()
	# connectedFilter.SetExtractionModeToAllRegions()
	# connectedFilter.ColorRegionsOn()
	connectedFilter.SetExtractionModeToClosestPointRegion()
	connectedFilter.SetClosestPoint(bifPoint)
	key_point_list = {}

	for key, surface in surfaces.items():
		clipBoxes_ = {}
		clipPlanes_ = {}
		boundaryCaps_ = {}
		key_point_list_ = {}

		kdTree = vtk.vtkKdTreePointLocator()
		kdTree.SetDataSet(centerlines[key])

		iD = kdTree.FindClosestPoint(bifPoint_list[key])
		kdTree.Update()

		bif_point_ = list(centerlines[key].GetPoint(iD))
		bif_point_tangent_ = list(centerlines[key].GetPointData().GetArray("FrenetTangent").GetTuple(iD))
		bif_point_normal_ = list(centerlines[key].GetPointData().GetArray("FrenetNormal").GetTuple(iD))
		bif_point_binormal_ = list(centerlines[key].GetPointData().GetArray("FrenetBinormal").GetTuple(iD))

		bif_point_dict = {"coordinate": bif_point_, "tangent": bif_point_tangent_, "normal": bif_point_normal_, "binormal": bif_point_binormal_}
		key_point_list_.update({"BifurcationPoint": bif_point_dict})

		iD = kdTree.FindClosestPoint(start_point)
		kdTree.Update()

		start_point_ = list(centerlines[key].GetPoint(iD))
		start_point_tangent_ = list(centerlines[key].GetPointData().GetArray("FrenetTangent").GetTuple(iD))
		start_point_normal_ = list(centerlines[key].GetPointData().GetArray("FrenetNormal").GetTuple(iD))
		start_point_binormal_ = list(centerlines[key].GetPointData().GetArray("FrenetBinormal").GetTuple(iD))

		start_point_dict = {"coordinate": start_point_, "tangent": start_point_tangent_, "normal": start_point_normal_, "binormal": start_point_binormal_}
		key_point_list_.update({"ICA": start_point_dict})

		clip_result = clip_polydata_by_box(surface, start_point_, start_point_tangent_, start_point_normal_, start_point_binormal_, size=[15,15,1], capping=capping)

		# perform lcc everytime after clipping to guarantee clean result
		connectedFilter.SetInputData(clip_result["clipped_surface"])
		connectedFilter.Update()
		surface.DeepCopy(connectedFilter.GetOutput())

		clipBoxes_.update({"ICA": clip_result["clip_box"]})
		clipPlanes_.update({"ICA": clip_result["clip_plane"]})
		connectedFilter_cap = vtk.vtkConnectivityFilter()
		connectedFilter_cap.SetExtractionModeToClosestPointRegion()
		connectedFilter_cap.SetClosestPoint([start_point_[i] + start_point_tangent_[i] for i in range(len(start_point_))])
		connectedFilter_cap.SetInputData(clip_result["boundary_cap"])
		connectedFilter_cap.Update()
		start_cap = vtk.vtkPolyData()
		start_cap.DeepCopy(connectedFilter_cap.GetOutput())
		boundaryCaps_.update({"ICA": start_cap})

		for i in range(len(end_points)):
			if i == 0:
				outlet_key = "ACA"
			else:
				outlet_key = "MCA"

			kdTree = vtk.vtkKdTreePointLocator()
			kdTree.SetDataSet(centerlines[key])
			iD = kdTree.FindClosestPoint(end_points[i])
			kdTree.Update()

			end_point_ = list(centerlines[key].GetPoint(iD))
			end_point_tangent_ = list(centerlines[key].GetPointData().GetArray("FrenetTangent").GetTuple(iD))
			end_point_normal_ = list(centerlines[key].GetPointData().GetArray("FrenetNormal").GetTuple(iD))
			end_point_binormal_ = list(centerlines[key].GetPointData().GetArray("FrenetBinormal").GetTuple(iD))

			end_point_dict = {"coordinate": end_point_, "tangent": end_point_tangent_, "normal": end_point_normal_, "binormal": end_point_binormal_}
			key_point_list_.update({outlet_key: start_point_dict})

			clip_result = clip_polydata_by_box(surface, end_point_, end_point_tangent_, end_point_normal_, end_point_binormal_, size=[10,10,1], capping=capping)
			# perform lcc everytime after clipping to guarantee clean result
			connectedFilter.SetInputData(clip_result["clipped_surface"])
			connectedFilter.Update()
			surface.DeepCopy(connectedFilter.GetOutput())

			clipBoxes_.update({outlet_key: clip_result["clip_box"]})
			clipPlanes_.update({outlet_key: clip_result["clip_plane"]})
			connectedFilter_cap = vtk.vtkConnectivityFilter()
			connectedFilter_cap.SetExtractionModeToClosestPointRegion()
			connectedFilter_cap.SetClosestPoint([end_point_[i] - end_point_tangent_[i] for i in range(len(end_point_))])
			connectedFilter_cap.SetInputData(clip_result["boundary_cap"])
			connectedFilter_cap.Update()
			end_cap = vtk.vtkPolyData()
			end_cap.DeepCopy(connectedFilter_cap.GetOutput())
			# end_cap = clip_result["boundary_cap"]
			boundaryCaps_.update({outlet_key: end_cap})

		clipBoxes.update({key: clipBoxes_})
		clipPlanes.update({key: clipPlanes_})
		boundaryCaps.update({key: boundaryCaps_})
		surfaces_clipped.update({key: surface})
		key_point_list.update({key: key_point_list_})

	for key,centerline in centerlines.items():
		kdTree = vtk.vtkKdTreePointLocator()
		kdTree.SetDataSet(centerlines[key])
		iD = kdTree.FindClosestPoint(start_point)
		kdTree.Update()

		start_point_ = list(centerlines[key].GetPoint(iD))
		start_point_tangent_ = list(centerlines[key].GetPointData().GetArray("FrenetTangent").GetTuple(iD))
		start_point_normal_ = list(centerlines[key].GetPointData().GetArray("FrenetNormal").GetTuple(iD))
		start_point_binormal_ = list(centerlines[key].GetPointData().GetArray("FrenetBinormal").GetTuple(iD))

		clip_result = clip_polydata_by_box(centerline, start_point_, start_point_tangent_, start_point_normal_, start_point_binormal_, size=[15,15,1])
		centerline = clip_result["clipped_surface"]

		for i in range(len(end_ids)):
			kdTree = vtk.vtkKdTreePointLocator()
			kdTree.SetDataSet(centerlines[key])
			iD = kdTree.FindClosestPoint(end_points[i])
			kdTree.Update()

			end_point_ = list(centerlines[key].GetPoint(iD))
			end_point_tangent_ = list(centerlines[key].GetPointData().GetArray("FrenetTangent").GetTuple(iD))
			end_point_normal_ = list(centerlines[key].GetPointData().GetArray("FrenetNormal").GetTuple(iD))
			end_point_binormal_ = list(centerlines[key].GetPointData().GetArray("FrenetBinormal").GetTuple(iD))

			clip_result = clip_polydata_by_box(centerline, end_point_, end_point_tangent_, end_point_normal_, end_point_binormal_, size=[10,10,1])
			centerline = clip_result["clipped_surface"]

		connectedFilter.SetInputData(centerline)
		connectedFilter.Update()
		centerline.DeepCopy(connectedFilter.GetOutput())
		centerlines_clipped.update({key: centerline})

	# output
	vtpWriter = vtk.vtkXMLPolyDataWriter()

	for key, value in centerlines_clipped.items():
		vtpWriter.SetFileName(os.path.join(case_dir,key,"centerline_clipped.vtp"))
		vtpWriter.SetInputData(value)
		vtpWriter.Update()

	stlWriter = vtk.vtkSTLWriter()

	for key, value in surfaces_clipped.items():
		stlWriter.SetFileName(os.path.join(case_dir,key,"surface_clipped.stl"))
		stlWriter.SetInputData(value)
		stlWriter.Update()

	for key, value in clipBoxes.items():
		for key_, value_ in value.items():
			stlWriter.SetFileName(os.path.join(case_dir,key,"clip_box_" + key_ + ".stl"))
			stlWriter.SetInputData(value_)
			stlWriter.Update()

	for key, value in clipPlanes.items():
		for key_, value_ in value.items():
			stlWriter.SetFileName(os.path.join(case_dir,key,"clip_plane_" + key_ + ".stl"))
			stlWriter.SetInputData(value_)
			stlWriter.Update()

	for key, value in boundaryCaps.items():
		for key_, value_ in value.items():
			stlWriter.SetFileName(os.path.join(case_dir,key,"boundary_cap_" + key_ + ".stl"))
			stlWriter.SetInputData(value_)
			stlWriter.Update()

	inletKeys = ["ICA","ACA","MCA"]

	if stl_concat:
		for phase in phases:
			if not os.path.exists(os.path.join(case_dir,phase,"surface_clipped.stl")):
				continue

			stl_text = open(os.path.join(case_dir,phase,"surface_clipped.stl")).read()
			stl_text = stl_text.replace("ascii","vessel")
			stl_text = stl_text.replace("Visualization Toolkit generated SLA File", "vessel")

			os.remove(os.path.join(case_dir,phase,"surface_capped.stl"))

			output_stl_object = open(os.path.join(case_dir,phase,"surface_capped.stl"),"a")
			output_stl_object.write(stl_text)

			for inletKey in inletKeys:
				stl_text = open(os.path.join(case_dir,phase,"boundary_cap_" + inletKey + ".stl")).read()
				stl_text = stl_text.replace("ascii",inletKey)
				stl_text = stl_text.replace("Visualization Toolkit generated SLA File",inletKey)
				output_stl_object.write(stl_text)
			output_stl_object.close()

	# output keypoint as json file
	for phase in phases:
		if not os.path.exists(os.path.join(case_dir,phase)):
				continue

		with open(os.path.join(case_dir,phase,"inlets.json"),"w") as fp:
			json.dump(key_point_list[phase], fp,indent=4)

def main():
	# # for comparison data
	# for case in os.listdir(data_dir)[0:]:
	# 	print(case)
	# 	execute(os.path.join(data_dir,case,"3DRA"))

	# 	exit()

	# for followup data
	phases = ["baseline","baseline-post","12months","followup"]

	dataList = os.listdir(data_dir)[:]
	# dataList = ["ChanSP"]

# 	dataList = ["LingKW_baseline-post","WongYK_followup"]

	for case in dataList:
		print(case)
		# phases = [case.split("_")[1]]
		# case = case.split("_")[0]

		for phase in phases:
			if batch_centerline:
				if phase == "followup":
					# continue
					centerlineCalculation(os.path.join(data_dir,case,phase),relaxation=smooth_relaxation_CBCT)
				else:
					# continue
					centerlineCalculation(os.path.join(data_dir,case,phase),relaxation=smooth_relaxation_3DRA)

		if batch_clip:
			normalizeVessels(os.path.join(data_dir,case))

if __name__=="__main__":
	main()