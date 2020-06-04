import os
import csv
import glob
import heapq
import numpy as np
import vtk

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
		self.parent = vtk.vtkRenderWindowInteractor()
		if(parent is not None):
			self.parent = parent

		self.AddObserver("KeyPressEvent",self.keyPressEvent)
		self.ActiveSelection = 0 # 0 refers to min point, 1 refers to max point

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
				print "To locate max point"
				self.parent.GetRenderWindow().Render()
			else:
				self.ActiveSelection = 0
				self.textActor.SetInput("Current selection: min point (red)")
				print "To locate min point"
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

def main():
	data_folder = "E:/Dr Simon Yu/CFD intracranial/data/stent/batch2/CheungSH"

	timePoints = ['baseline','baseline-post','12months','followup']
	minPoint = (0,0,0)
	maxPoint = (0,0,0)
	for timePoint in timePoints:
		if os.path.exists(os.path.join(data_folder,timePoint
			)):
			centerline = os.path.join(data_folder,timePoint,"centerline-probe-1.vtp")
			surface = os.path.join(data_folder,timePoint,"surface_capped.stl")

			centerlineReader = vtk.vtkXMLPolyDataReader()
			centerlineReader.SetFileName(centerline)
			centerlineReader.Update()
			centerline = centerlineReader.GetOutput()
			centerline.GetCellData().SetScalars(centerline.GetCellData().GetArray(2));
			centerline.GetPointData().SetScalars(centerline.GetPointData().GetArray("Abscissas"));

			surfaceReader = vtk.vtkSTLReader()
			surfaceReader.SetFileName(surface)
			surfaceReader.Update()
			surface = surfaceReader.GetOutput()

			lut = vtk.vtkLookupTable()
			# lut.SetNumberOfTableValues(3);
			lut.Build()

			centerlineMapper = vtk.vtkPolyDataMapper()
			centerlineMapper.SetInputData(centerline)
			centerlineMapper.SetScalarRange(0, centerline.GetPointData().GetScalars().GetMaxNorm());
			centerlineMapper.SetLookupTable(lut);
			centerlineMapper.SetScalarModeToUsePointData()

			surfaceMapper = vtk.vtkPolyDataMapper()
			surfaceMapper.SetInputData(surface)

			scalarBar = vtk.vtkScalarBarActor()
			scalarBar.SetLookupTable(centerlineMapper.GetLookupTable());
			scalarBar.SetTitle("Abscissas")
			scalarBar.SetNumberOfLabels(4)
			scalarBar.SetWidth(0.08)
			scalarBar.SetHeight(0.6)
			scalarBar.SetPosition(0.9,0.1)

			# auto find the smallest radius point
			radius = centerline.GetPointData().GetArray("Radius")
			minIdx = vtkArrayMin(radius)
			maxIdx = vtkArrayMax(radius)

			# build kd tree to locate the nearest point
			# Create kd tree
			kDTree = vtk.vtkKdTreePointLocator()
			kDTree.SetDataSet(centerline)
			kDTree.BuildLocator()

			minSource = vtk.vtkSphereSource()
			if timePoint == 'baseline':
				minSource.SetCenter(centerline.GetPoint(minIdx))
			else:
				# Find the closest points to picked point
				iD = kDTree.FindClosestPoint(minPoint)
	 
				# Get the coordinates of the closest point
				closestPoint = kDTree.GetDataSet().GetPoint(iD)
				minSource.SetCenter(closestPoint)
			minSource.SetRadius(0.3);
			minMapper = vtk.vtkPolyDataMapper()
			minMapper.SetInputConnection(minSource.GetOutputPort());
			minActor = vtk.vtkActor()
			minActor.SetMapper(minMapper);
			minActor.GetProperty().SetColor((1.0,0.0,0.0))

			maxSource = vtk.vtkSphereSource()
			if timePoint == 'baseline':
				maxSource.SetCenter(centerline.GetPoint(maxIdx))
			else:
				# Find the closest points to picked point
				iD = kDTree.FindClosestPoint(maxPoint)
	 
				# Get the coordinates of the closest point
				closestPoint = kDTree.GetDataSet().GetPoint(iD)
				maxSource.SetCenter(closestPoint)
			maxSource.SetRadius(0.3);
			maxMapper = vtk.vtkPolyDataMapper()
			maxMapper.SetInputConnection(maxSource.GetOutputPort());
			maxActor = vtk.vtkActor()
			maxActor.SetMapper(maxMapper);
			maxActor.GetProperty().SetColor((0.0,1.0,0.0))

			centerlineActor = vtk.vtkActor()
			centerlineActor.SetMapper(centerlineMapper)
			
			surfaceActor = vtk.vtkActor()
			surfaceActor.SetMapper(surfaceMapper)       
			surfaceActor.GetProperty().SetOpacity(0.3)

			# text actor
			usageTextActor = vtk.vtkTextActor()
			usageTextActor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
			usageTextActor.GetPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
			usageTextActor.SetPosition([0.001, 0.05])
			usageTextActor.SetInput("Tab: Switch max/min point\nSpace: Locate max/min point\nEnter/Close Window: Process")

			currentSelectionTextActor = vtk.vtkTextActor()
			currentSelectionTextActor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
			currentSelectionTextActor.GetPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
			currentSelectionTextActor.SetPosition([0.25, 0.1])
			currentSelectionTextActor.SetInput("Current selection: min point (red)")

			renderer = vtk.vtkRenderer()
			renderer.AddActor(centerlineActor)
			renderer.AddActor(surfaceActor)
			renderer.AddActor(minActor)
			renderer.AddActor(maxActor)
			renderer.AddActor2D(scalarBar);
			renderer.AddActor(usageTextActor)
			renderer.AddActor(currentSelectionTextActor)

			renderWindow = vtk.vtkRenderWindow()
			renderWindow.AddRenderer(renderer)

			renderWindowInteractor = vtk.vtkRenderWindowInteractor()
			renderWindowInteractor.SetRenderWindow(renderWindow)
			mystyle = MyInteractorStyle(renderWindowInteractor)
			mystyle.SetMaxSphereSource(maxSource)
			mystyle.SetMinSphereSource(minSource)
			mystyle.SetCenterline(centerline)
			mystyle.SetSelectionTextActor(currentSelectionTextActor)
			renderWindowInteractor.SetInteractorStyle(mystyle)

			renderWindow.SetSize(1024,780); #(width, height)
			renderWindow.Render()
			renderWindowInteractor.Start()

			minPoint = minSource.GetCenter()
			maxPoint = maxSource.GetCenter()

			# compute degree of stenosis
			minIdx = kDTree.FindClosestPoint(minPoint)
			minRadius = centerline.GetPointData().GetArray("Radius").GetTuple(minIdx)[0]
			maxIdx = kDTree.FindClosestPoint(maxPoint)
			maxRadius = centerline.GetPointData().GetArray("Radius").GetTuple(maxIdx)[0]
			DoS = (1-minRadius/maxRadius)*100

			print("{}: min radius = {} mm, Degree of stenosis = {} %".format(timePoint,minRadius,DoS))

if __name__=="__main__":
	main()