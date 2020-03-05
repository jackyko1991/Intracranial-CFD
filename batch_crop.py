import os
import shutil
import vtk
import math

data_dir = "D:/Dr_Simon_Yu/CFD_intracranial/data/comparison"
binary_path = "D:/Dr_Simon_Yu/CFD_intracranial/code/cxx/Vessel-Centerline-Extraction/build/Release/CenterlineExtraction.exe"
dist_from_defect = 20

def clip_polydata_by_box(polydata, point, tangent, normal, binormal, size=[10,10,1]):
	"""Clip the input polydata with given box shape and direction

	Parameters:
	polydata (polydata): Input polydata
	point (array): Coordinate of clipping box center
	normal (array): Direction tangent of the clipping box (major axis)
	normal (array): Direction normal of the clipping box
	normal (array): Direction binormal of the clipping box
	size (array): Size of the clipping box (default: [10,10,1])

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

def execute(working_dir):
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
	# os.system(command)

	# crop equal distance from the defected zone
	# load defected zone coordinate
	defected_point = open(os.path.join(working_dir,"defected_point.fcsv"),"r").readlines()[3]
	defected_point = defected_point.split(",")[1:4]
	defected_point = [float(i) for i in defected_point]

	# load the centerline file
	reader = vtk.vtkXMLPolyDataReader()
	reader.SetFileName(os.path.join(working_dir,"centerline.vtp"))
	reader.Update()
	centerline = reader.GetOutput()

	kdTree = vtk.vtkKdTreePointLocator()
	kdTree.SetDataSet(centerline)
	iD = kdTree.FindClosestPoint(defected_point)

	end_id = centerline.GetNumberOfPoints()-1

	defected_point_absc = centerline.GetPointData().GetArray("Abscissas").GetComponent(iD,0)

	# find the start point
	start_id = 0
	start_point_absc = 0
	start_point = [0,0,0]
	start_point_tangent = [1,0,0]
	start_point_normal = [0,1,0]
	start_point_binormal = [0,0,1]

	for i in range(centerline.GetNumberOfPoints()-1):
		if defected_point_absc - centerline.GetPointData().GetArray("Abscissas").GetComponent(i,0)<dist_from_defect:
			break
		else:
			start_id = i
			start_point_absc = centerline.GetPointData().GetArray("Abscissas").GetComponent(i,0)
			start_point = list(centerline.GetPoints().GetPoint(i))
			start_point_tangent = list(centerline.GetPointData().GetArray("FrenetTangent").GetTuple(i))
			start_point_normal = list(centerline.GetPointData().GetArray("FrenetNormal").GetTuple(i))
			start_point_binormal = list(centerline.GetPointData().GetArray("FrenetBinormal").GetTuple(i))

	print(start_id,start_point,start_point_tangent,start_point_normal,start_point_binormal)

	clipBoxes = [] 
	clipPlanes = []
	surface, clipBox, clipPlane = clip_polydata_by_box(surface, start_point, start_point_tangent, start_point_normal, start_point_binormal)
	centerline, _ , _ = clip_polydata_by_box(centerline, start_point, start_point_tangent, start_point_normal, start_point_binormal)
	clipBoxes.append(clipBox)
	clipPlanes.append(clipPlane)

	# find end points
	# split the polydata by centerline id
	splitter = vtk.vtkThreshold()
	splitter.SetInputData(centerline)

	splitted_centerlines = []
	for i in range(int(centerline.GetCellData().GetArray("CenterlineIds").GetRange()[1])+1):
		splitter.ThresholdBetween(i,i)
		splitter.SetInputArrayToProcess(0,0,0,vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS,"CenterlineIds")
		splitter.Update()

		splitted_centerline = vtk.vtkPolyData()
		splitted_centerline.DeepCopy(splitter.GetOutput())
		splitted_centerlines.append(splitted_centerline)

	end_ids = []
	end_points = []
	end_points_tangent = []
	end_points_normal = []
	end_points_binormal = []

	for splitted_centerline in splitted_centerlines:
		end_id = splitted_centerline.GetNumberOfPoints()-1
		end_point_absc = splitted_centerline.GetPointData().GetArray("Abscissas").GetComponent(splitted_centerline.GetNumberOfPoints()-1,0)
		end_point = list(splitted_centerline.GetPoints().GetPoint(splitted_centerline.GetNumberOfPoints()-1))
		end_point_tangent = list(splitted_centerline.GetPointData().GetArray("FrenetTangent").GetTuple(splitted_centerline.GetNumberOfPoints()-1))
		end_point_normal = list(splitted_centerline.GetPointData().GetArray("FrenetNormal").GetTuple(splitted_centerline.GetNumberOfPoints()-1))
		end_point_binormal = list(splitted_centerline.GetPointData().GetArray("FrenetBinormal").GetTuple(splitted_centerline.GetNumberOfPoints()-1))

		for i in range(splitted_centerline.GetNumberOfPoints()):
			if splitted_centerline.GetPointData().GetArray("Abscissas").GetComponent(i,0) - start_point_absc > 2*dist_from_defect:
				end_ids.append(end_id)
				end_points.append(end_point)
				end_points_tangent.append(end_point_tangent)
				end_points_normal.append(end_point_normal)
				end_points_binormal.append(end_point_binormal)
				break
			else:
				end_id = i
				end_point_absc = splitted_centerline.GetPointData().GetArray("Abscissas").GetComponent(i,0)
				end_point = list(splitted_centerline.GetPoints().GetPoint(i))
				end_point_tangent = list(splitted_centerline.GetPointData().GetArray("FrenetTangent").GetTuple(i))
				end_point_normal = list(splitted_centerline.GetPointData().GetArray("FrenetNormal").GetTuple(i))
				end_point_binormal = list(splitted_centerline.GetPointData().GetArray("FrenetBinormal").GetTuple(i))

			if i == splitted_centerline.GetNumberOfPoints()-1:
				end_ids.append(end_id)
				end_points.append(end_point)
				end_points_tangent.append(end_point_tangent)
				end_points_normal.append(end_point_normal)
				end_points_binormal.append(end_point_binormal)

	for i in range(len(end_ids)):
		print(end_ids[i],end_points[i],end_points_tangent[i],end_points_normal[i],end_points_binormal[i])
		surface, clipBox, clipPlane = clip_polydata_by_box(surface, end_points[i], end_points_tangent[i], end_points_normal[i], end_points_binormal[i])
		centerline, _ , _ = clip_polydata_by_box(centerline, end_points[i], end_points_tangent[i], end_points_normal[i], end_points_binormal[i])

		clipBoxes.append(clipBox)
		clipPlanes.append(clipPlane)

	# connected component calculation on surface and centerline
	connectedFilter = vtk.vtkConnectivityFilter()
	# connectedFilter.SetExtractionModeToAllRegions()
	# connectedFilter.ColorRegionsOn()
	connectedFilter.SetExtractionModeToClosestPointRegion()
	connectedFilter.SetClosestPoint(defected_point)
	connectedFilter.SetInputData(surface)
	connectedFilter.Update()
	surface.DeepCopy(connectedFilter.GetOutput())

	# output
	vtpWriter = vtk.vtkXMLPolyDataWriter()
	vtpWriter.SetFileName(os.path.join(working_dir,"surface_clipped.vtp"))
	vtpWriter.SetInputData(surface)
	vtpWriter.Update()

	connectedFilter.SetInputData(centerline)
	connectedFilter.Update()
	centerline.DeepCopy(connectedFilter.GetOutput())

	vtpWriter.SetFileName(os.path.join(working_dir,"centerline_clipped.vtp"))
	vtpWriter.SetInputData(centerline)
	vtpWriter.Update()

	for i in range(len(clipBoxes)):
		writer.SetFileName(os.path.join(working_dir,"clip_box_" + str(i) + ".stl"))
		writer.SetInputData(clipBoxes[i])
		writer.Update()

		writer.SetFileName(os.path.join(working_dir,"clip_plane_" + str(i) + ".stl"))
		writer.SetInputData(clipPlanes[i])
		writer.Update()

	# prepare surface for mesh generation


def main():
	for case in os.listdir(data_dir)[0:]:
		print(case)
		execute(os.path.join(data_dir,case,"3DRA"))
		exit()

if __name__=="__main__":
	main()