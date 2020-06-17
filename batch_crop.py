import os
import shutil
import vtk
import math
import numpy as np

data_dir = "Z:/data/intracranial/followup/medical"
binary_path = "D:/projects/CFD_intracranial/cxx/Vessel-Centerline-Extraction/build/Release/CenterlineExtraction.exe"
dist_from_bif = 20

def clip_polydata_by_box(polydata, point, tangent, normal, binormal, size=[10,10,1]):
	"""Clip the input polydata with given box shape and direction

	Parameters:
	polydata (polydata): Input polydata
	point (array): Coordinate of clipping box center
	normal (array): Direction tangent of the clipping box (major axis)
	normal (array): Direction normal of the clipping box
	normal (array): Direction binormal of the clipping box
	size (array): Size of the clipping box (default: [10,10,1], note that actual clip box size will be half of that input, bug to be fixed)

	Returns:
	vtkPolyData: Clipped polydata
	vtkPolyData: Clipping box
	vtkPolyData: Clipping plane
	"""

	# create a clipping box widget
	clipWidget = vtk.vtkBoxWidget()
	transform = vtk.vtkTransform()

	transform.Translate(point)	
	w = math.atan(math.sqrt(tangent[0]**2+tangent[1]**2)/tangent[2])*180/3.14
	transform.RotateWXYZ(w, -tangent[1], tangent[0],0)
	transform.Scale(size)
	
	# print(transform.GetMatrix())

	clipBox = vtk.vtkCubeSource()
	transformFilter = vtk.vtkTransformPolyDataFilter()
	transformFilter.SetInputConnection(clipBox.GetOutputPort())
	transformFilter.SetTransform(transform)
	transformFilter.Update()

	clipBoxPolyData = vtk.vtkPolyData()
	clipBoxPolyData.DeepCopy(transformFilter.GetOutput())

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

	# clipping plane
	point1 = []
	point2 = []
	origin = []
	
	for i in range(3):
		point1.append(point[i]-normal[i]*math.sqrt(2)/2*size[0])
		point2.append(point[i]+normal[i]*math.sqrt(2)/2*size[0])
		origin.append(point[i]+binormal[i]*math.sqrt(2)/2*size[0])

	planeSource = vtk.vtkPlaneSource()
	planeSource.SetResolution(10,10)
	planeSource.SetOrigin(origin)
	planeSource.SetPoint1(point1)
	planeSource.SetPoint2(point2)
	planeSource.Update()

	clipPlanePolyData = vtk.vtkPolyData()
	clipPlanePolyData.DeepCopy(planeSource.GetOutput())
	return polydata, clipBoxPolyData, clipPlanePolyData

def centerlineCalculation(working_dir):
	if not os.path.exists(os.path.join(working_dir,"surface.vtk")):
		return

	# convert vtk to stl
	reader = vtk.vtkGenericDataObjectReader()
	reader.SetFileName(os.path.join(working_dir,"surface.vtk"))
	reader.Update()
	surface = reader.GetOutput()
	
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
		if not os.path.exists(os.path.join(case_dir,phase,"surface.vtk")):
			continue
		reader = vtk.vtkGenericDataObjectReader()
		reader.SetFileName(os.path.join(case_dir,phase,"surface.vtk"))
		reader.Update()
		surface = reader.GetOutput()
		surfaces.update({phase: surface})

	# get the bifurcation point from baseline data
	# split the polydata by centerline id
	splitter = vtk.vtkThreshold()
	splitter.SetInputData(centerlines['baseline'])
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
		endPoint1_absc_list.update({key: ACA.GetPointData().GetArray("Abscissas").GetComponent(ACA.GetNumberOfPoints()-1,0)})
		endPoint1_id_list.update({key: ACA.GetNumberOfPoints()-1})

		splitter.ThresholdBetween(3,3)
		splitter.Update()
		MCA = splitter.GetOutput()
		endPoint2_absc_list.update({key: ACA.GetPointData().GetArray("Abscissas").GetComponent(MCA.GetNumberOfPoints()-1,0)})
		endPoint2_id_list.update({key: ACA.GetNumberOfPoints()-1})

	# get the start point coordinate
	start_id = 0
	start_point_absc = 0
	start_point = [0,0,0]
	start_point_tangent = [1,0,0]
	start_point_normal = [0,1,0]
	start_point_binormal = [0,0,1]

	for i in range(centerlines["baseline"].GetNumberOfPoints()):
		if (bifPoint_absc_list["baseline"] - centerlines["baseline"].GetPointData().GetArray("Abscissas").GetComponent(i,0) < min(bifPoint_absc_list.values())) and \
			(bifPoint_absc_list["baseline"] - centerlines["baseline"].GetPointData().GetArray("Abscissas").GetComponent(i,0) < dist_from_bif):
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

		for i in range(splitted_centerline.GetNumberOfPoints()):
			end_id = splitted_centerline.GetNumberOfPoints()-1
			end_point_absc = splitted_centerline.GetPointData().GetArray("Abscissas").GetComponent(splitted_centerline.GetNumberOfPoints()-1,0)
			end_point = list(splitted_centerline.GetPoints().GetPoint(splitted_centerline.GetNumberOfPoints()-1))
			end_point_tangent = list(splitted_centerline.GetPointData().GetArray("FrenetTangent").GetTuple(splitted_centerline.GetNumberOfPoints()-1))
			end_point_normal = list(splitted_centerline.GetPointData().GetArray("FrenetNormal").GetTuple(splitted_centerline.GetNumberOfPoints()-1))
			end_point_binormal = list(splitted_centerline.GetPointData().GetArray("FrenetBinormal").GetTuple(splitted_centerline.GetNumberOfPoints()-1))

			if (splitter.GetOutput().GetPointData().GetArray("Abscissas").GetComponent(i,0) < min(endPoint1_absc_list.values())) and \
				(splitter.GetOutput().GetPointData().GetArray("Abscissas").GetComponent(i,0)-bifPoint_absc_list["baseline"] < dist_from_bif):
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

	print("end_points",end_points)

	# clip the surfaces
	clipBoxes = {}
	clipPlanes = {}
	surfaces_clipped = {}
	centerlines_clipped = {}
	# connected component calculation on surface and centerline
	connectedFilter = vtk.vtkConnectivityFilter()
	# connectedFilter.SetExtractionModeToAllRegions()
	# connectedFilter.ColorRegionsOn()
	connectedFilter.SetExtractionModeToClosestPointRegion()
	connectedFilter.SetClosestPoint(bifPoint)

	for key, surface in surfaces.items():
		clipBoxes_ = []
		clipPlanes_ = []

		surface, clipBox, clipPlane = clip_polydata_by_box(surface, start_point, start_point_tangent, start_point_normal, start_point_binormal, size=[20,20,0.5])

		clipBoxes_.append(clipBox)
		clipPlanes_.append(clipPlane)

		for i in range(len(end_ids)):
			# print(end_ids[i],end_points[i],end_points_tangent[i],end_points_normal[i],end_points_binormal[i])
			surface, clipBox, clipPlane = clip_polydata_by_box(surface, end_points[i], end_points_tangent[i], end_points_normal[i], end_points_binormal[i], size=[20,20,0.5])
			clipBoxes_.append(clipBox)
			clipPlanes_.append(clipPlane)

		connectedFilter.SetInputData(surface)
		connectedFilter.Update()
		surface.DeepCopy(connectedFilter.GetOutput())
		clipBoxes.update({key: clipBoxes_})
		clipPlanes.update({key: clipPlanes_})
		surfaces_clipped.update({key: surface})

	for centerline in centerlines.values():
		centerline, _ , _ = clip_polydata_by_box(centerline, start_point, start_point_tangent, start_point_normal, start_point_binormal)

		for i in range(len(end_ids)):
			# print(end_ids[i],end_points[i],end_points_tangent[i],end_points_normal[i],end_points_binormal[i])
			centerline, _ , _ = clip_polydata_by_box(centerline, end_points[i], end_points_tangent[i], end_points_normal[i], end_points_binormal[i])

		connectedFilter.SetInputData(centerline)
		connectedFilter.Update()
		centerline.DeepCopy(connectedFilter.GetOutput())
		centerlines_clipped.update({key: centerline})

	# output
	vtpWriter = vtk.vtkXMLPolyDataWriter()

	for key, value in surfaces_clipped.items():
		vtpWriter.SetFileName(os.path.join(case_dir,key,"surface_clipped.vtp"))
		vtpWriter.SetInputData(value)
		vtpWriter.Update()

	for key, value in centerlines_clipped.items():
		vtpWriter.SetFileName(os.path.join(case_dir,key,"centerline_clipped.vtp"))
		vtpWriter.SetInputData(value)
		vtpWriter.Update()

	writer = vtk.vtkSTLWriter()
	for key, value in clipBoxes.items():
		for i in range(len(value)):
			writer.SetFileName(os.path.join(case_dir,key,"clip_box_" + str(i) + ".stl"))
			writer.SetInputData(value[i])
			writer.Update()

	for key, value in clipPlanes.items():
		for i in range(len(value)):
			writer.SetFileName(os.path.join(case_dir,key,"clip_plane_" + str(i) + ".stl"))
			writer.SetInputData(value[i])
			writer.Update()

def main():
	# # for comparison data
	# for case in os.listdir(data_dir)[0:]:
	# 	print(case)
	# 	execute(os.path.join(data_dir,case,"3DRA"))

	# 	exit()

	# for followup data
	phases = ["baseline","baseline-post","12months","followup"]

	dataList = os.listdir(data_dir)[0:]

	for case in dataList:
		# for phase in phases:
			# centerlineCalculation(os.path.join(data_dir,case,phase))

		normalizeVessels(os.path.join(data_dir,case))

		exit()

if __name__=="__main__":
	main()