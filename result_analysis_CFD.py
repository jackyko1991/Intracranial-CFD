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
 
	def SetMinSphereSource(self,minSphereSource):
		self.minSphereSource = minSphereSource

	def SetMaxSphereSource(self,maxSphereSource):
		self.maxSphereSource = maxSphereSource

	def SetCenterline(self,centerline):
		self.centerline = centerline

	# def leftButtonPressEvent(self,obj,event):
	# 	clickPos = self.GetInteractor().GetEventPosition()

	# 	picker = vtk.vtkPropPicker()
	# 	picker.Pick(clickPos[0], clickPos[1], 0, self.GetDefaultRenderer())

	# 	# get the new
	# 	self.NewPickedActor = picker.GetActor()

	# 	# If something was selected
	# 	if self.NewPickedActor:
	# 		# If we picked something before, reset its property
	# 		if self.LastPickedActor:
	# 			self.LastPickedActor.GetProperty().DeepCopy(self.LastPickedProperty)

	# 		# Save the property of the picked actor so that we can
	# 		# restore it next time
	# 		self.LastPickedProperty.DeepCopy(self.NewPickedActor.GetProperty())
	# 		# Highlight the picked actor by changing its properties
	# 		self.NewPickedActor.GetProperty().SetColor(1.0, 0.0, 0.0)
	# 		self.NewPickedActor.GetProperty().SetDiffuse(1.0)
	# 		self.NewPickedActor.GetProperty().SetSpecular(0.0)

	# 		# save the last picked actor
	# 		self.LastPickedActor = self.NewPickedActor

	# 	self.OnLeftButtonDown()
	# 	return

	def keyPressEvent(self,obj,event):
		key = self.parent.GetKeySym()
		if key == 'Tab':
			if self.ActiveSelection == 0:
				self.ActiveSelection = 1
				print "To locate max point"
			else:
				self.ActiveSelection = 0
				print "To locate min point"
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
	output_file = "E:/Dr Simon Yu/CFD intracranial/data/result-stent-2.csv"
	data_folder = "E:/Dr Simon Yu/CFD intracranial/data/stent"
	group = 'stent'

	# create result CSV file with header
	with open(output_file, 'wb') as output:
		fieldnames = ['Patient','Group','time point','branch',
		'radius mean(mm)','degree of stenosis(%)','radius min(mm)',
		'pressure mean(mmHg)','max pressure gradient(mmHg)','in/out pressure gradient(mmHg)',
		'velocity mean(ms^-1)','peak velocity(ms^-1)',
		'shear strain rate mean(Pas^-1)','peak shear strain rate(Pas^-1)',
		'vorticity mean(s^-1)','peak vorticity(s^-1)']
		writer = csv.DictWriter(output, fieldnames=fieldnames)
		writer.writeheader()

	for batch in os.listdir(data_folder):
		if os.path.isdir(os.path.join(data_folder,batch)):
			for patient in os.listdir(os.path.join(data_folder,batch)):
				timePoints = ['baseline','baseline-post','12months','followup']
				for timePoint in timePoints:
					omit_cases = ['ChanT','ChanWK','YeungSL','TsuiCY']
					if timePoint ==  'followup' and patient in omit_cases:
						break

					if os.path.exists(os.path.join(data_folder,batch,patient,timePoint)):
						for file in glob.glob(os.path.join(data_folder,batch,patient,timePoint,"*.csv")):
							print file
								
							column = readCSV(file)

							# write the result to CSV file
							row = []
							row.append(patient)
							row.append(group)
							row.append(timePoint)
							row.append(file[-5:-4])
							row.append(np.mean(np.asarray(column["Radius"]))) #radius mean
							row.append((1-np.min(column["Radius"])/np.max(column["Radius"]))*100) # degree of stenosis
							row.append(np.min(column["Radius"])) #radius min
							row.append(np.mean(np.asarray(column["Absolute_Pressure"]))*0.00750061683) #absolute pressure
							row.append((np.mean(heapq.nlargest(10, column["Absolute_Pressure"]))-np.mean(heapq.nsmallest(10, column["Absolute_Pressure"])))*0.00750061683) # max pressure gradient
							row.append((np.mean(column["Absolute_Pressure"][0:10])-np.mean(column["Absolute_Pressure"][-10:-1]))*0.00750061683) #in/out pressure gradient
							velx = np.asarray(column["Velocity:0"])
							vely = np.asarray(column["Velocity:1"])
							velz = np.asarray(column["Velocity:2"])
							vel = np.sqrt(np.square(velx)+np.square(vely)+np.square(velz))
							row.append(np.mean(vel)) #velocity mean
							row.append(np.mean(heapq.nlargest(10, vel))) # peak velocity

							row.append(np.mean(np.asarray(column["Shear_Strain_Rate"]))) #shear strain rate mean
							row.append(np.mean(heapq.nlargest(10, np.asarray(column["Shear_Strain_Rate"])))) # peak strain rate mean

							vorx = np.asarray(column["Vorticity:0"])
							vory = np.asarray(column["Vorticity:1"])
							vorz = np.asarray(column["Vorticity:2"])
							vor = np.sqrt(np.square(vorx)+np.square(vory)+np.square(vorz))
							row.append(np.mean(vor)) #vorticity mean
							row.append(np.mean(heapq.nlargest(10, vor))) # peak vorticity

							CSVwrite(output_file,row)

if __name__=="__main__":
	main()