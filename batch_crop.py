import os
import shutil
import vtk
import math

data_dir = "D:/Dr_Simon_Yu/CFD_intracranial/data/comparison"
binary_path = "D:/Dr_Simon_Yu/CFD_intracranial/code/cxx/Vessel-Centerline-Extraction/build/Release/CenterlineExtraction.exe"
dist_from_defect = 20

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

	# print(defected_point)
	# print(centerline)
	# print(centerline.GetPointData().GetArray("Abscissas").GetComponent(iD,0))

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

	# start_point_normal = [(-1*i) for i in start_point_normal]

	print(start_id,start_point,start_point_tangent,start_point_normal,start_point_binormal)

	# create a clipping box widget
	clipWidget = vtk.vtkBoxWidget()
	transform = vtk.vtkTransform()

	transform.Translate(start_point[0],start_point[1],start_point[2])	
	w = math.atan(math.sqrt(start_point_tangent[0]**2+start_point_tangent[1]**2)/start_point_tangent[2])*180/3.14
	transform.RotateWXYZ(w, -start_point_tangent[1], start_point_tangent[0],0)
	transform.Scale(10,10,1)
	
	print(transform.GetMatrix())

	clipBox = vtk.vtkCubeSource()
	transformFilter = vtk.vtkTransformPolyDataFilter()
	transformFilter.SetInputConnection(clipBox.GetOutputPort())
	transformFilter.SetTransform(transform)
	transformFilter.Update()

	writer.SetFileName(os.path.join(working_dir,"clip_box.stl"))
	writer.SetInputData(transformFilter.GetOutput())
	writer.Update()

	clipWidget.SetTransform(transform)
	clipFunction = vtk.vtkPlanes()
	clipWidget.GetPlanes(clipFunction)

	clipper = vtk.vtkClipPolyData()
	clipper.SetClipFunction(clipFunction)
	clipper.SetInputData(surface)
	clipper.GenerateClippedOutputOn()
	clipper.SetValue(0.0)
	clipper.Update()

	surface.DeepCopy(clipper.GetOutput())

	writer.SetFileName(os.path.join(working_dir,"surface_clipped.stl"))
	writer.SetInputData(surface)
	writer.Update()

	# clipping plane
	point1 = []
	point2 = []
	origin = []
	
	for i in range(3):
		point1.append(start_point[i]-start_point_normal[i]*math.sqrt(2)/2*5)
		point2.append(start_point[i]+start_point_normal[i]*math.sqrt(2)/2*5)
		origin.append(start_point[i]+start_point_binormal[i]*math.sqrt(2)/2*5)

	planeSource = vtk.vtkPlaneSource()
	planeSource.SetResolution(10,10)
	planeSource.SetOrigin(origin)
	planeSource.SetPoint1(point1)
	planeSource.SetPoint2(point2)
	planeSource.Update()

	box = vtk.vtkPlanes()
	points = vtk.vtkPoints()
	points.InsertNextPoint(point1)
	points.InsertNextPoint(point2)
	points.InsertNextPoint(origin)

	writer.SetFileName(os.path.join(working_dir,"clip_plane_1.stl"))
	writer.SetInputData(planeSource.GetOutput())
	writer.Update()

	exit()

	print(origin)
	print(point1)
	print(point2)
	

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

	# # clip the surface
	# plane = vtk.vtkPlane()
	# plane.SetOrigin(start_point)
	# plane.SetNormal(start_point_normal)

	# clipper = vtk.vtkClipPolyData()
	# clipper.SetClipFunction(plane)
	# clipper.SetInputData(surface)
	# # clipper.Update()

	# # surface.DeepCopy(clipper.GetOutput())

	# clipper.InsideOutOn()
	# for i in range(len(end_ids)):
	# 	if i == 0:
	# 		continue
	# 	plane.SetOrigin(end_points[i])
	# 	plane.SetNormal(end_points_normal[i])

	# 	clipper.SetClipFunction(plane)
	# 	clipper.SetInputData(surface)
	# 	clipper.Update()

	# 	surface.DeepCopy(clipper.GetOutput())

	# writer.SetFileName(os.path.join(working_dir,"surface_clipped.stl"))
	# writer.SetInputData(surface)
	# writer.Update()

def main():
	for case in os.listdir(data_dir)[1:]:
		print(case)
		execute(os.path.join(data_dir,case,"3DRA"))
		exit()

if __name__=="__main__":
	main()