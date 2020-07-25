import os
import csv
import glob
import heapq
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import pandas as pd
import matplotlib.pyplot as plt
import math

def readCSV(path):
	f = open(path, 'rb')
	reader = csv.reader(f)
	headers = reader.next()
	column = {}
	for h in headers:
		column[h] = []
	for row in reader:
		for h, v in zip(headers, row):
			if v == "-1.#IND":
				v = 0 
			column[h].append(float(v))
	return column

def CSVwrite(CSVpath,row):
	csvfile = open(CSVpath,'a')
	csv_writer = csv.writer(csvfile)
	csv_writer.writerow(row)
	csvfile.close()

def vtkArrayMin(vtkArray):
	arrayMax = vtkArray.GetDataTypeMax()
	idx = 0
	for i in range(vtkArray.GetNumberOfTuples()):
		if vtkArray.GetTuple(i)[0] < arrayMax:
			arrayMax = vtkArray.GetTuple(i)[0]
			idx = i
	return idx

def vtkArrayMax(vtkArray):
	arrayMin = vtkArray.GetDataTypeMin()
	idx = 0
	for i in range(vtkArray.GetNumberOfTuples()):
		if vtkArray.GetTuple(i)[0] > arrayMin:
			arrayMin = vtkArray.GetTuple(i)[0]
			idx = i
	return idx

class MyInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
 
	def __init__(self,parent=None):
		# intractorStyleTrackballCamera does not inherit from render window interactor, nned to port manually
		# self.parent = self.GetInteractor()
		# print self.parent
		self.parent = vtk.vtkRenderWindowInteractor()
		if(parent is not None):
			self.parent = parent

		# self.AddObserver("LeftButtonPressEvent",self.leftButtonPressEvent)
		self.AddObserver("KeyPressEvent",self.keyPressEvent)
		self.ActiveSelection = 0 # 0 refers to min point, 1 refers to max point
 
		# self.LastPickedActor = None
		# self.LastPickedProperty = vtk.vtkProperty()

		self.minSphereSource = None
		self.maxSphereSource = None
		self.centerline = None
		self.textActor = None
 
	def SetMinSphereSource(self,minSphereSource):
		self.minSphereSource = minSphereSource

	def SetMaxSphereSource(self,maxSphereSource):
		self.maxSphereSource = maxSphereSource

	def SetCenterline(self,centerline):
		self.centerline = centerline

	def SetSelectionTextActor(self,textActor):
		self.textActor = textActor

	def keyPressEvent(self,obj,event):
		key = self.parent.GetKeySym()
		if key == 'Tab':
			if self.ActiveSelection == 0:
				self.ActiveSelection = 1
				self.textActor.SetInput("Current selection: max point (green)")
				print("To locate max point")
				self.parent.GetRenderWindow().Render()
			else:
				self.ActiveSelection = 0
				self.textActor.SetInput("Current selection: min point (red)")
				print("To locate min point")
				self.parent.GetRenderWindow().Render()
		if key == "space":
			self.parent.GetPicker().Pick(self.parent.GetEventPosition()[0],
					self.parent.GetEventPosition()[1],
					0,  # always zero.
					self.parent.GetRenderWindow().GetRenderers().GetFirstRenderer())
			picked = self.parent.GetPicker().GetPickPosition();

			# Create kd tree
			kDTree = vtk.vtkKdTreePointLocator()
			kDTree.SetDataSet(self.centerline)
			kDTree.BuildLocator()

			# Find the closest points to picked point
			iD = kDTree.FindClosestPoint(picked)
 
			# Get the coordinates of the closest point
			closestPoint = kDTree.GetDataSet().GetPoint(iD)

			if self.ActiveSelection == 0:
				self.minSphereSource.SetCenter(closestPoint)
			else:
				self.maxSphereSource.SetCenter(closestPoint)
			self.parent.GetRenderWindow().Render()
		if key == "Return":
			# Close the window
  			self.parent.GetRenderWindow().Finalize()
  			# Stop the interactor
  			self.parent.TerminateApp()

		return

def plot_centerline_result(centerline, array_names):
	# extract ica
	thresholdFilter = vtk.vtkThreshold()
	thresholdFilter.ThresholdBetween(1,1)
	thresholdFilter.SetInputData(centerline)
	thresholdFilter.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "CenterlineIds_average")
	thresholdFilter.Update()

	x = vtk_to_numpy(thresholdFilter.GetOutput().GetPointData().GetArray("Abscissas_average"))
	x = [(value - x[0]) for value in x]
	
	fig, axs = plt.subplots(len(array_names),1)
	fig.suptitle("CFD result")

	for i in range(len(array_names)):
		y = vtk_to_numpy(thresholdFilter.GetOutput().GetPointData().GetArray(array_names[i]))

		if len(y.shape) > 1:
			if y.shape[1] == 3:
				y = [math.sqrt(value[0]**2 + value[1]**2 +value[2]**2 ) for value in y]

		print(y)
		if len(array_names) == 1:
			ax = axs
		else:
			ax = axs[i]

		ax.plot(x,y)
		ax.set_xlabel("Abscissas (mm)")

		if array_names[i] == "Radius_average":
			ylabel = "Radius (mm)"
			ymax = 2.5
		elif array_names[i] == "U_average":
			ylabel = "Velocity (ms^-1)"
			ymax = 2.0
		elif array_names[i] == "p(mmHg)_average":
			ylabel = "Pressure (mmHg)"
			ymax = 80
		elif array_names[i] == "vorticity_average":
			ylabel = "vorticity (s^-1)"
			ymax = 1000

		ax.set_ylabel(ylabel)
		ax.set_xlim(x[0],x[-1])
		ax.set_ylim(0,ymax)

	plt.show()


def centerline_probe_result(centerline_file,vtk_file_list, output_dir):
	# read centerline
	centerlineReader = vtk.vtkXMLPolyDataReader()
	centerlineReader.SetFileName(centerline_file)
	centerlineReader.Update()
	centerline = centerlineReader.GetOutput()

	# read vtk files
	vtk_files = []
	centerlines = []

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	if not os.path.exists(os.path.join(output_dir,"centerlines")):
		os.makedirs(os.path.join(output_dir,"centerlines"))

	# average filter
	averageFilter = vtk.vtkTemporalStatistics()

	for file_name in vtk_file_list:
		reader = vtk.vtkUnstructuredGridReader()
		reader.SetFileName(file_name)
		reader.Update() 

		geomFilter = vtk.vtkGeometryFilter()
		geomFilter.SetInputData(reader.GetOutput())
		geomFilter.Update()

		# scale up the CFD result
		transform = vtk.vtkTransform()
		transform.Scale(1000,1000,1000)

		transformFilter = vtk.vtkTransformPolyDataFilter()
		transformFilter.SetInputData(geomFilter.GetOutput())
		transformFilter.SetTransform(transform)
		transformFilter.Update()

		vtk_files.append(transformFilter.GetOutput())

		interpolator = vtk.vtkPointInterpolator()
		interpolator.SetSourceData(transformFilter.GetOutput())
		interpolator.SetInputData(centerline)
		interpolator.Update()

		# convert to desired unit
		converter = vtk.vtkArrayCalculator()
		converter.SetInputData(interpolator.GetOutput())
		converter.AddScalarArrayName("p")
		converter.SetFunction("p * 921 * 0.0075") # 921 = mu/nu = density of blood, 0.0075 converts from Pascal to mmHg
		converter.SetResultArrayName("p(mmHg)")
		converter.Update()

		# output the probe centerline
		centerline_output_path = os.path.join(
			os.path.dirname(centerline_file),
			output_dir,
			"centerlines",
			"centerline_probe_{}.vtp".format(os.path.split(file_name)[1].split("_")[1].split(".")[0]) 
			)

		centerlines.append(converter.GetOutput())
		averageFilter.SetInputData(converter.GetOutput())
		averageFilter.Update()

		writer = vtk.vtkXMLPolyDataWriter()
		writer.SetInputData(converter.GetOutput())
		writer.SetFileName(centerline_output_path)
		writer.Update()

	centerline_output_path = os.path.join(
				os.path.dirname(centerline_file),
				output_dir,
				"centerlines",
				"centerline_probe_avg.vtp" 
				)

	writer = vtk.vtkXMLPolyDataWriter()
	writer.SetInputData(averageFilter.GetOutput())
	writer.SetFileName(centerline_output_path)
	writer.Update()

	# plot result
	centerline_average = averageFilter.GetOutput()
	print(centerline_average)

	plot_centerline_result(centerline_average,["Radius_average","U_average","p(mmHg)_average","vorticity_average"])

	# os.system("pause")
	exit()

	return

def result_analysis(case_dir, minPoint=(0,0,0), maxPoint=(0,0,0)):
	centerline = os.path.join(case_dir, "centerline_clipped.vtp")
	surface = os.path.join(case_dir, "surface_capped.stl")
	output_dir = os.path.join(case_dir,"CFD_OpenFOAM_result")

	# centerlineReader = vtk.vtkXMLPolyDataReader()
	# centerlineReader.SetFileName(centerline)
	# centerlineReader.Update()
	# centerline = centerlineReader.GetOutput()
	# centerline.GetCellData().SetScalars(centerline.GetCellData().GetArray(2));
	# centerline.GetPointData().SetScalars(centerline.GetPointData().GetArray("Abscissas"));
	# print(centerline)

	# surfaceReader = vtk.vtkSTLReader()
	# surfaceReader.SetFileName(surface)
	# surfaceReader.Update()
	# surface = surfaceReader.GetOutput()

	# lut = vtk.vtkLookupTable()
	# # lut.SetNumberOfTableValues(3);
	# lut.Build()

	# # Fill in a few known colors, the rest will be generated if needed
	# # lut.SetTableValue(0, 1.0000, 0     , 0, 1);  
	# # lut.SetTableValue(1, 0.0000, 1.0000, 0.0000, 1);
	# # lut.SetTableValue(2, 0.0000, 0.0000, 1.0000, 1); 

	# centerlineMapper = vtk.vtkPolyDataMapper()
	# centerlineMapper.SetInputData(centerline)
	# centerlineMapper.SetScalarRange(0, centerline.GetPointData().GetScalars().GetMaxNorm());
	# centerlineMapper.SetLookupTable(lut);
	# centerlineMapper.SetScalarModeToUsePointData()

	# surfaceMapper = vtk.vtkPolyDataMapper()
	# surfaceMapper.SetInputData(surface)

	# scalarBar = vtk.vtkScalarBarActor()
	# scalarBar.SetLookupTable(centerlineMapper.GetLookupTable());
	# scalarBar.SetTitle("Abscissas");
	# scalarBar.SetNumberOfLabels(4);
	# scalarBar.SetWidth(0.08)
	# scalarBar.SetHeight(0.6)
	# scalarBar.SetPosition(0.9,0.1)

	# # auto find the smallest radius point
	# radius = centerline.GetPointData().GetArray("Radius")

	# # build kd tree to locate the nearest point
	# # Create kd tree
	# kDTree = vtk.vtkKdTreePointLocator()
	# kDTree.SetDataSet(centerline)
	# kDTree.BuildLocator()

	# minSource = vtk.vtkSphereSource()
	# if minPoint == (0,0,0):
	# 	minIdx = vtkArrayMin(radius)
	# 	closestPoint = centerline.GetPoint(minIdx)
	# else:
	# 	# Find the closest point to the picked point
	# 	iD = kDTree.FindClosestPoint(minPoint)

	# 	# Get the id of the closest point
	# 	closestPoint = kDTree.GetDataSet().GetPoint(iD)

	# minSource.SetCenter(closestPoint)
	# minSource.SetRadius(0.3);
	# minMapper = vtk.vtkPolyDataMapper()
	# minMapper.SetInputConnection(minSource.GetOutputPort());
	# minActor = vtk.vtkActor()
	# minActor.SetMapper(minMapper);
	# minActor.GetProperty().SetColor((1.0,0.0,0.0))

	# maxSource = vtk.vtkSphereSource()
	# if maxPoint == (0,0,0):
	# 	maxIdx = vtkArrayMin(radius)
	# 	closestPoint = centerline.GetPoint(maxIdx)
	# else:
	# 	# Find the closest point to the picked point
	# 	iD = kDTree.FindClosestPoint(maxPoint)

	# 	# Get the id of the closest point
	# 	closestPoint = kDTree.GetDataSet().GetPoint(iD)

	# maxSource.SetCenter(closestPoint)
	# maxSource.SetRadius(0.3);
	# maxMapper = vtk.vtkPolyDataMapper()
	# maxMapper.SetInputConnection(minSource.GetOutputPort());
	# maxActor = vtk.vtkActor()
	# maxActor.SetMapper(maxMapper);
	# maxActor.GetProperty().SetColor((1.0,0.0,0.0))

	# centerlineActor = vtk.vtkActor()
	# centerlineActor.SetMapper(centerlineMapper)
	
	# surfaceActor = vtk.vtkActor()
	# surfaceActor.SetMapper(surfaceMapper)       
	# surfaceActor.GetProperty().SetOpacity(0.3)

	# # text actor
	# usageTextActor = vtk.vtkTextActor()
	# usageTextActor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
	# usageTextActor.GetPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
	# usageTextActor.SetPosition([0.001, 0.05])
	# usageTextActor.SetInput("Tab: Switch max/min point\nSpace: Locate max/min point\nEnter/Close Window: Process")

	# currentSelectionTextActor = vtk.vtkTextActor()
	# currentSelectionTextActor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
	# currentSelectionTextActor.GetPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
	# currentSelectionTextActor.SetPosition([0.25, 0.1])
	# currentSelectionTextActor.SetInput("Current selection: min point (red)")

	# renderer = vtk.vtkRenderer()
	# renderer.AddActor(centerlineActor)
	# renderer.AddActor(surfaceActor)
	# renderer.AddActor(minActor)
	# renderer.AddActor(maxActor)
	# renderer.AddActor2D(scalarBar)
	# renderer.AddActor(usageTextActor)
	# renderer.AddActor(currentSelectionTextActor)

	# renderWindow = vtk.vtkRenderWindow()
	# renderWindow.AddRenderer(renderer)

	# renderWindowInteractor = vtk.vtkRenderWindowInteractor()
	# renderWindowInteractor.SetRenderWindow(renderWindow)
	# mystyle = MyInteractorStyle(renderWindowInteractor)
	# mystyle.SetMaxSphereSource(maxSource)
	# mystyle.SetMinSphereSource(minSource)
	# mystyle.SetCenterline(centerline)
	# mystyle.SetSelectionTextActor(currentSelectionTextActor)
	# renderWindowInteractor.SetInteractorStyle(mystyle)

	# renderWindow.SetSize(1024,780); #(width, height)
	# renderWindow.Render()
	# renderWindowInteractor.Start()

	# minPoint = minSource.GetCenter()
	# maxPoint = maxSource.GetCenter()

	# # compute degree of stenosis
	# minIdx = kDTree.FindClosestPoint(minPoint)
	# minRadius = centerline.GetPointData().GetArray("Radius").GetTuple(minIdx)[0]
	# maxIdx = kDTree.FindClosestPoint(maxPoint)
	# maxRadius = centerline.GetPointData().GetArray("Radius").GetTuple(maxIdx)[0]
	# DoS = (1-minRadius/maxRadius)*100

	# print("{}: min radius = {} mm, Degree of stenosis = {} %".format(timePoint,minRadius,DoS))

	# load last 5 time points and take average
	results_vtk = []

	for time in range(0,201,10):
		results_vtk.append(os.path.join(case_dir,"CFD_OpenFOAM", "VTK","OpenFoam_" + str(time)+".vtk"))
	
	centerline_probe_result(os.path.join(case_dir,"centerline_clipped.vtp"),results_vtk[-5:],output_dir)

	return {}
	
def main():
	group = "medical"

	output_file = "D:/Projects/intracranial/data/followup/result.csv".format(group)
	data_folder = "D:/Projects/intracranial/data/followup/{}".format(group)

	# create result dataframe
	field_names = ['patient','group','time point',
		'radius mean(mm)','degree of stenosis(%)','radius min(mm)',
		'pressure mean(mmHg)','max pressure gradient(mmHg)','in/out pressure gradient(mmHg)',
		'velocity mean(ms^-1)','peak velocity(ms^-1)',
		'shear strain rate mean(Pas^-1)','peak shear strain rate(Pas^-1)',
		'vorticity mean(s^-1)','peak vorticity(s^-1)']

	result_df = pd.DataFrame(columns = field_names)
	# timePoints = ['baseline','baseline-post','12months','followup']
	timePoints = ["baseline"]
	# for case in os.listdir(data_folder):
	for case in ["ChowLM"]:
		for timePoint in timePoints:
			if not os.path.exists(os.path.join(data_folder,case,timePoint)):
				continue
			
			row = {"patient": case, "group": group, "time point": timePoint}
			result = result_analysis(os.path.join(data_folder,case,timePoint))
			
			result_df = result_df.append(pd.Series(row),ignore_index=True)
			result_df.to_csv(output_file,index=False)

			exit()

if __name__=="__main__":
	main()