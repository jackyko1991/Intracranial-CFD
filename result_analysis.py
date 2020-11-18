import os
import csv
import glob
import heapq
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import math
import json
from tqdm import tqdm
from scipy.interpolate import interp1d

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

def plot_centerline_result(centerline, array_names, result_path, dev_result_path ,minPoint=(0,0,0),bifurcationPoint=(0,0,0)):
	# extract ica
	thresholdFilter = vtk.vtkThreshold()
	thresholdFilter.ThresholdBetween(1,1)
	thresholdFilter.SetInputData(centerline)
	thresholdFilter.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "CenterlineIds_average")
	thresholdFilter.Update()

	if thresholdFilter.GetOutput().GetNumberOfPoints() == 0:
		return

	x = vtk_to_numpy(thresholdFilter.GetOutput().GetPointData().GetArray("Abscissas_average"))
	x = [(value - x[0]) for value in x]
	
	kDTree = vtk.vtkKdTreePointLocator()
	kDTree.SetDataSet(centerline)
	kDTree.BuildLocator()

	# get the abscissas of bifurcation point
	if bifurcationPoint != (0,0,0):
		iD = kDTree.FindClosestPoint(bifurcationPoint)
		# Get the id of the closest point
		bifPointAbscissas = kDTree.GetDataSet().GetPointData().GetArray("Abscissas_average").GetTuple(iD)[0] - \
			vtk_to_numpy(thresholdFilter.GetOutput().GetPointData().GetArray("Abscissas_average"))[0]

	# get the abscissas of min point
	if minPoint != (0,0,0):
		iD = kDTree.FindClosestPoint(minPoint)
		# Get the id of the closest point
		minPointAbscissas = kDTree.GetDataSet().GetPointData().GetArray("Abscissas_average").GetTuple(iD)[0] - \
			vtk_to_numpy(thresholdFilter.GetOutput().GetPointData().GetArray("Abscissas_average"))[0]

	fig, axs = plt.subplots(len(array_names),1)
	fig.suptitle("CFD result")
	fig.set_size_inches(10,8)

	fig2, axs2 = plt.subplots(len(array_names),1)
	fig2.suptitle("CFD result derivatives")
	fig2.set_size_inches(10,8)

	for i in range(len(array_names)):
		for lineId in range(int(centerline.GetCellData().GetArray("CenterlineIds_average").GetMaxNorm())):
			thresholdFilter.ThresholdBetween(lineId,lineId)
			thresholdFilter.Update()

			x = vtk_to_numpy(thresholdFilter.GetOutput().GetPointData().GetArray("Abscissas_average"))
			x = [(value - x[0]) for value in x]

			y = vtk_to_numpy(thresholdFilter.GetOutput().GetPointData().GetArray(array_names[i]))

			if len(y.shape) > 1:
				if y.shape[1] == 3:
					y = [math.sqrt(value[0]**2 + value[1]**2 +value[2]**2 ) for value in y]
					
			if len(array_names) == 1:
				ax = axs
				ax2 = axs2
			else:
				ax = axs[i]
				ax2 = axs2[i]

			order = np.argsort(x)
			xs = np.array(x)[order]
			ys = np.array(y)[order]

			unique, index = np.unique(xs, axis=-1, return_index=True)
			xs = xs[index]
			ys = ys[index]

			f = interp1d(xs,ys,kind="cubic")
			xs = np.linspace(0, np.amax(x), num=200, endpoint=True)
			ys = f(xs)

			dys = np.gradient(ys, xs)

			ax.plot(xs,ys)
			ax2.plot(xs,dys)

		if array_names[i] == "Radius_average":
			ylabel = "Radius (mm)"
			ymin=0
			ymax = 5
		elif array_names[i] == "U_average":
			ylabel = "Velocity (ms^-1)"
			ymin=0
			ymax = 3
		elif array_names[i] == "p(mmHg)_average":
			ylabel = "Pressure (mmHg)"
			ymin=0
			ymax = 180
		elif array_names[i] == "vorticity_average":
			ylabel = "Vorticity (s^-1)"
			ymin=0
			ymax = 4000
		elif array_names[i] == "Curvature_average":
			ylabel = "Curvature"
			ymin = -1.5
			ymax = 1.5
		elif array_names[i] == "Torsion_average":
			ylabel = "Torsion"
			ymin = -100000
			ymax = 100000

		if bifurcationPoint !=(0,0,0):
			ax.axvline(x=bifPointAbscissas,ymin=ymin,ymax=ymax,linestyle ="--",color='m')
			ax2.axvline(x=bifPointAbscissas,ymin=ymin,ymax=ymax,linestyle ="--",color='m')

		if minPoint !=(0,0,0):
			ax.axvline(x=minPointAbscissas,ymin=ymin,ymax=ymax,linestyle ="--",color='c')
			ax2.axvline(x=minPointAbscissas,ymin=ymin,ymax=ymax,linestyle ="--",color='c')

		ax.set_ylabel(ylabel)
		ax2.set_ylabel(ylabel)
		if i == (len(array_names)-1):
			ax.set_xlabel("Abscissas (mm)")
			ax2.set_xlabel("Abscissas (mm)")
		else:
			ax.set_xticklabels([])
			ax2.set_xticklabels([])
		ax.set_xlim(x[0],x[-1])
		ax.set_ylim(ymin,ymax)
		ax2.set_xlim(x[0],x[-1])
		ax2.set_ylim(ymin,ymax)
		
	# save the plot 
	fig.savefig(result_path,dpi=100)
	fig.clf()

	fig2.savefig(dev_result_path,dpi=100)
	fig2.clf()

	plt.close("all")

def centerline_probe_result(centerline_file,vtk_file_list, output_dir,minPoint=(0,0,0), bifurcationPoint=(0,0,0)):
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
		# get the first element pressure
		try:
			first_point_pressure = interpolator.GetOutput().GetPointData().GetArray("p").GetValue(0)
		except:
			first_point_pressure = 120

		converter = vtk.vtkArrayCalculator()
		converter.SetInputData(interpolator.GetOutput())
		converter.AddScalarArrayName("p")
		converter.SetFunction("120 + (p - {}) * 921 * 0.0075".format(first_point_pressure)) # 921 = mu/nu = density of blood, 0.0075 converts from Pascal to mmHg, offset 120mmHg at ica
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
	plot_result_path = os.path.join(os.path.dirname(centerline_file),output_dir,"result.png")
	dev_plot_result_path = os.path.join(os.path.dirname(centerline_file),output_dir,"result_dev.png")
	plot_centerline_result(
		averageFilter.GetOutput(),
		["Radius_average","U_average","p(mmHg)_average","vorticity_average","Curvature_average","Torsion_average"], 
		plot_result_path,
		dev_plot_result_path,
		minPoint = minPoint,
		bifurcationPoint = bifurcationPoint)

	# extract ica
	thresholdFilter = vtk.vtkThreshold()
	thresholdFilter.ThresholdBetween(1,1)
	thresholdFilter.SetInputData(averageFilter.GetOutput())
	thresholdFilter.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "CenterlineIds_average")
	thresholdFilter.Update()

	if thresholdFilter.GetOutput().GetNumberOfPoints() == 0:
		tqdm.write("Centerline file {} does not contain suitable number of CenterlineIds".format(centerline_file))
		return {
			'radius mean(mm)': "NA",
			'radius min(mm)': "NA",
			'pressure mean(mmHg)': "NA",
			'max pressure gradient(mmHg)': "NA",
			'in/out pressure gradient(mmHg)': "NA",
			'velocity mean(ms^-1)': "NA",
			'peak velocity(ms^-1)': "NA",
			'vorticity mean(s^-1)': "NA",
			'peak vorticity(s^-1)': "NA"
			}

	# compute result values
	abscissas = vtk_to_numpy(thresholdFilter.GetOutput().GetPointData().GetArray("Abscissas_average"))
	radius = vtk_to_numpy(thresholdFilter.GetOutput().GetPointData().GetArray("Radius_average"))
	pressure = vtk_to_numpy(thresholdFilter.GetOutput().GetPointData().GetArray("p(mmHg)_average"))
	pressure_gradient = np.diff(pressure)/np.diff(abscissas)
	velocity = vtk_to_numpy(thresholdFilter.GetOutput().GetPointData().GetArray("U_average"))
	velocity = [math.sqrt(value[0]**2 + value[1]**2 +value[2]**2 ) for value in velocity]
	vorticity = vtk_to_numpy(thresholdFilter.GetOutput().GetPointData().GetArray("vorticity_average"))
	vorticity = [math.sqrt(value[0]**2 + value[1]**2 +value[2]**2 ) for value in vorticity]

	return_value = {
		'radius mean(mm)': np.mean(radius),
		'radius min(mm)': np.min(radius),
		'pressure mean(mmHg)': np.mean(pressure),
		'max pressure gradient(mmHg)': np.mean(heapq.nlargest(5, pressure_gradient)),
		'in/out pressure gradient(mmHg)': np.mean(pressure[0:5]) - np.mean(pressure[-5:]),
		'velocity mean(ms^-1)': np.mean(velocity),
		'peak velocity(ms^-1)': np.mean(heapq.nlargest(5, velocity)),
		'vorticity mean(s^-1)': np.mean(vorticity),
		'peak vorticity(s^-1)': np.mean(heapq.nlargest(5, vorticity))
	}

	# moving variance matrix
	mv_fields = [
		"Radius_average",
		"U_average",
		"p(mmHg)_average",
		"vorticity_average",
		"Curvature_average",
		"Torsion_average"
		]
	mv_windows = np.arange(3,10,2)
	plot_result_path = os.path.join(os.path.dirname(centerline_file),output_dir,"moving_variance.png")
	dev_plot_result_path = os.path.join(os.path.dirname(centerline_file),output_dir,"moving_variance_dev.png")

	mv_matrix_df, mv_dy_matrix_df = moving_variance_matrix(averageFilter.GetOutput(),mv_fields,mv_windows,
		minPoint = minPoint,
		bifurcationPoint = bifurcationPoint,
		result_path = plot_result_path,
		dev_result_path=dev_plot_result_path)

	return return_value, mv_matrix_df, mv_dy_matrix_df

def rolling_window(a, window):
	pad = np.zeros(len(a.shape), dtype=np.int32)
	pad[-1] = window-1
	pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
	a = np.pad(a, pad,mode='edge')
	shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
	strides = a.strides + (a.strides[-1],)

	return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def moving_variance_matrix(centerline,fields, windows, minPoint=(0,0,0), bifurcationPoint=(0,0,0), result_path="", dev_result_path=""):
	# create input array from centerline
	thresholdFilter = vtk.vtkThreshold()
	thresholdFilter.SetInputData(centerline)
	thresholdFilter.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "CenterlineIds_average")
	thresholdFilter.Update()

	fig, axs = plt.subplots(len(fields),1)
	fig.suptitle("CFD Moving Variance")
	fig.set_size_inches(10,8)

	fig2, axs2 = plt.subplots(len(fields),1)
	fig2.suptitle("CFD Derivative Moving Variance")
	fig2.set_size_inches(10,8)

	# build kd tree to locate the nearest point
	# Create kd tree
	kDTree = vtk.vtkKdTreePointLocator()
	kDTree.SetDataSet(centerline)
	kDTree.BuildLocator()

	# get the abscissas of bifurcation point
	if bifurcationPoint != (0,0,0):
		iD = kDTree.FindClosestPoint(bifurcationPoint)
		# Get the id of the closest point
		bifPointAbscissas = kDTree.GetDataSet().GetPointData().GetArray("Abscissas_average").GetTuple(iD)[0] - \
			vtk_to_numpy(thresholdFilter.GetOutput().GetPointData().GetArray("Abscissas_average"))[0]

	# get the abscissas of min point
	if minPoint != (0,0,0):
		iD = kDTree.FindClosestPoint(minPoint)
		# Get the id of the closest point
		minPointAbscissas = kDTree.GetDataSet().GetPointData().GetArray("Abscissas_average").GetTuple(iD)[0] - \
			vtk_to_numpy(thresholdFilter.GetOutput().GetPointData().GetArray("Abscissas_average"))[0]

	col_name = ["branch","window"] + fields
	pmv_df = pd.DataFrame(columns=col_name)
	col_name_dy = ["branch","window"] + [y+"_dev" for y in fields]
	pmv_dy_df = pd.DataFrame(columns=col_name_dy)

	for lineId in range(int(centerline.GetCellData().GetArray("CenterlineIds_average").GetMaxNorm())):
		a_x = []
		a_y = []
		a_dy = []

		thresholdFilter.ThresholdBetween(lineId,lineId)
		thresholdFilter.Update()

		# need at least 3 points to perform interpolation
		if thresholdFilter.GetOutput().GetNumberOfPoints() < 4:
			continue

		for i in range(len(fields)):
			x = vtk_to_numpy(thresholdFilter.GetOutput().GetPointData().GetArray("Abscissas_average"))
			x = [(value - x[0]) for value in x]

			y = vtk_to_numpy(thresholdFilter.GetOutput().GetPointData().GetArray(fields[i]))

			if len(y.shape) > 1:
				if y.shape[1] == 3:
					y = [math.sqrt(value[0]**2 + value[1]**2 +value[2]**2 ) for value in y]
					
			order = np.argsort(x)
			xs = np.array(x)[order]
			ys = np.array(y)[order]

			unique, index = np.unique(xs, axis=-1, return_index=True)
			xs = xs[index]
			ys = ys[index]

			f = interp1d(xs,ys,kind="cubic")
			xnew = np.linspace(0, np.amax(x), num=50, endpoint=True)
			ynew = f(xnew)

			dy = np.gradient(ynew, xnew)

			a_x.append(xnew)
			a_y.append(ynew)
			a_dy.append(dy)

		a_x = np.array(a_x)
		a_y = np.array(a_y)
		a_dy = np.array(a_dy)

		for window in windows:
			mv = np.var(rolling_window(a_y, window) , axis=-1)
			mv_dy = np.var(rolling_window(a_dy, window) , axis=-1)

			pmv = np.amax(mv,axis=-1)
			pmv = np.concatenate(([lineId,window],pmv))
			pmv_df.loc[len(pmv_df)] = pmv

			pmv_dy = np.amax(mv_dy,axis=-1)
			pmv_dy = np.concatenate(([lineId,window],pmv_dy))
			pmv_dy_df.loc[len(pmv_dy_df)] = pmv_dy

			for i in range(len(fields)):
				# plot moving variance
				if len(fields) == 1:
					ax = axs
					ax2 = axs2
				else:
					ax = axs[i]
					ax2 = axs2[i]

				ax.plot(a_x[i,:],mv[i,:])
				ax2.plot(a_x[i,:],mv_dy[i,:])

				if fields[i] == "Radius_average":
					ylabel = "Radius (mm)"
					ymin = 0
					ymax = 0.5
				elif fields[i] == "U_average":
					ylabel = "Velocity (ms^-1)"
					ymin = 0
					ymax = 0.5
				elif fields[i] == "p(mmHg)_average":
					ylabel = "Pressure (mmHg)"
					ymin = 0
					ymax = 1e3
				elif fields[i] == "vorticity_average":
					ylabel = "Vorticity (s^-1)"
					ymin = 0
					ymax = 1e7
				elif fields[i] == "Curvature_average":
					ylabel = "Curvature"
					ymin = 0
					ymax = 1e0
				elif fields[i] == "Torsion_average":
					ylabel = "Torsion"
					ymin = 0
					ymax = 1e9

				if bifurcationPoint !=(0,0,0):
					ax.axvline(x=bifPointAbscissas,ymin=0,ymax=1,linestyle="--",color='m')
					ax2.axvline(x=bifPointAbscissas,ymin=0,ymax=1,linestyle="--",color='m')

				if minPoint !=(0,0,0):
					ax.axvline(x=minPointAbscissas,ymin=0,ymax=1,linestyle="--",color='c')
					ax2.axvline(x=minPointAbscissas,ymin=0,ymax=1,linestyle="--",color='c')

				ax.set_ylabel(ylabel)
				ax2.set_ylabel(ylabel)
				if i == (len(fields)-1):
					ax.set_xlabel("Abscissas (mm)")
					ax2.set_xlabel("Abscissas (mm)")
				else:
					ax.set_xticklabels([])
					ax2.set_xticklabels([])
				ax.set_xlim(x[0],x[-1])
				ax.set_ylim(ymin,ymax)
				ax2.set_xlim(x[0],x[-1])
				ax2.set_ylim(ymin,ymax)

	# save the plot 
	fig.savefig(result_path,dpi=100)
	fig.clf()

	fig2.savefig(dev_result_path,dpi=100)
	fig2.clf()

	plt.close("all")

	pmv_df = pmv_df.groupby(['window']).max()
	pmv_df = pmv_df.drop(columns=['branch'])

	pmv_dy_df = pmv_dy_df.groupby(['window']).max()
	pmv_dy_df = pmv_dy_df.drop(columns=['branch'])

	return pmv_df, pmv_dy_df

def probe_min_max_point(centerline_filename, surface_filename, minPoint=(0,0,0), maxPoint=(0,0,0)):
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

	# Fill in a few known colors, the rest will be generated if needed
	# lut.SetTableValue(0, 1.0000, 0     , 0, 1);  
	# lut.SetTableValue(1, 0.0000, 1.0000, 0.0000, 1);
	# lut.SetTableValue(2, 0.0000, 0.0000, 1.0000, 1); 

	centerlineMapper = vtk.vtkPolyDataMapper()
	centerlineMapper.SetInputData(centerline)
	centerlineMapper.SetScalarRange(0, centerline.GetPointData().GetScalars().GetMaxNorm());
	centerlineMapper.SetLookupTable(lut);
	centerlineMapper.SetScalarModeToUsePointData()

	surfaceMapper = vtk.vtkPolyDataMapper()
	surfaceMapper.SetInputData(surface)

	scalarBar = vtk.vtkScalarBarActor()
	scalarBar.SetLookupTable(centerlineMapper.GetLookupTable());
	scalarBar.SetTitle("Abscissas");
	scalarBar.SetNumberOfLabels(4);
	scalarBar.SetWidth(0.08)
	scalarBar.SetHeight(0.6)
	scalarBar.SetPosition(0.9,0.1)

	# auto find the smallest radius point
	radius = centerline.GetPointData().GetArray("Radius")

	# build kd tree to locate the nearest point
	# Create kd tree
	kDTree = vtk.vtkKdTreePointLocator()
	kDTree.SetDataSet(centerline)
	kDTree.BuildLocator()

	minSource = vtk.vtkSphereSource()
	if minPoint == (0,0,0):
		minIdx = vtkArrayMin(radius)
		closestPoint = centerline.GetPoint(minIdx)
	else:
		# Find the closest point to the picked point
		iD = kDTree.FindClosestPoint(minPoint)

		# Get the id of the closest point
		closestPoint = kDTree.GetDataSet().GetPoint(iD)

	minSource.SetCenter(closestPoint)
	minSource.SetRadius(0.3);
	minMapper = vtk.vtkPolyDataMapper()
	minMapper.SetInputConnection(minSource.GetOutputPort());
	minActor = vtk.vtkActor()
	minActor.SetMapper(minMapper);
	minActor.GetProperty().SetColor((1.0,0.0,0.0))

	maxSource = vtk.vtkSphereSource()
	if maxPoint == (0,0,0):
		maxIdx = vtkArrayMin(radius)
		closestPoint = centerline.GetPoint(maxIdx)
	else:
		# Find the closest point to the picked point
		iD = kDTree.FindClosestPoint(maxPoint)

		# Get the id of the closest point
		closestPoint = kDTree.GetDataSet().GetPoint(iD)

	maxSource.SetCenter(closestPoint)
	maxSource.SetRadius(0.3);
	maxMapper = vtk.vtkPolyDataMapper()
	maxMapper.SetInputConnection(minSource.GetOutputPort());
	maxActor = vtk.vtkActor()
	maxActor.SetMapper(maxMapper);
	maxActor.GetProperty().SetColor((1.0,0.0,0.0))

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
	renderer.AddActor2D(scalarBar)
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

	return DoS, minPoint, maxPoint

def result_analysis(case_dir, minPoint=(0,0,0), maxPoint=(0,0,0), probe=False):
	# load domain json file
	with open(os.path.join(case_dir,"domain.json")) as f:
		domain = json.load(f)

	centerline = os.path.join(case_dir, domain["centerline"]["filename"])
	surface = os.path.join(case_dir, domain["domain"]["filename"])
	output_dir = os.path.join(case_dir,"CFD_OpenFOAM_result")

	if probe:
		Dos, minPoint, maxPoint = probe_min_max_point(centerline,surface,minPoint,maxPoint)

	# load last 5 time points and take average
	results_vtk = []

	for time in range(0,2001,100):
		results_vtk.append(os.path.join(case_dir,"CFD_OpenFOAM", "VTK","OpenFOAM_" + str(time)+".vtk"))
	
	try:
		minPoint = tuple(domain["fiducial_0"]["coordinate"])
	except:
		minPoint = (0,0,0)

	return_value, mv_matrix_df, mv_dy_matrix_df = centerline_probe_result(
		os.path.join(case_dir,domain["centerline"]["filename"]),
		results_vtk[-5:],
		output_dir, 
		minPoint=minPoint,
		bifurcationPoint=domain["bifurcation_point"]["coordinate"]
		)

	return return_value, mv_matrix_df, mv_dy_matrix_df, minPoint, maxPoint
	
def main():
	group = "stent"
	probe = False

	output_file = "/mnt/DIIR-JK-NAS/data/intracranial/followup/result-{}.csv".format(group,group)
	data_folder = "/mnt/DIIR-JK-NAS/data/intracranial/followup/{}".format(group)

	# create result dataframe
	field_names = ['patient','group','time point',
		'radius mean(mm)','degree of stenosis(%)','radius min(mm)',
		'pressure mean(mmHg)','max pressure gradient(mmHg)','in/out pressure gradient(mmHg)',
		'velocity mean(ms^-1)','peak velocity(ms^-1)',
		'shear strain rate mean(Pas^-1)','peak shear strain rate(Pas^-1)',
		'vorticity mean(s^-1)','peak vorticity(s^-1)']

	result_df = pd.DataFrame(columns=field_names)
	# timePoints = ['baseline','baseline-post','12months','followup']
	# timePoints = ["baseline"]

	pbar = tqdm(os.listdir(data_folder))
	# pbar = tqdm(["ChowLM"])
	for case in pbar:
		pbar.set_description(case)
		minPoint = (0,0,0)
		maxPoint = (0,0,0)

		pbar2 = tqdm(timePoints)
		for timePoint in pbar2:
			if not os.path.exists(os.path.join(data_folder,case,timePoint)):
				continue
			pbar2.set_description(timePoint)

			row = {"patient": case, "group": group, "time point": timePoint}
			result, result_dy, minPoint, maxPoint = result_analysis(os.path.join(data_folder,case,timePoint),minPoint=minPoint,maxPoint=maxPoint,probe=probe)
			row.update(result)
			result_df = result_df.append(pd.Series(row),ignore_index=True)
	result_df.to_csv(output_file,index=False)

def main2():
	probe=False
	group="stenosis"

	# output_file = "/mnt/DIIR-JK-NAS/data/intracranial/data_30_30/result_EASIS_medical.csv"
	# data_folder = "/mnt/DIIR-JK-NAS/data/intracranial/data_30_30/stenosis/ESASIS_medical"

	output_file = "Z:/data/intracranial/data_30_30/result_EASIS_medical.csv"
	data_folder = "Z:/data/intracranial/data_30_30/stenosis/ESASIS_medical"

	output_file = "Z:/data/intracranial/data_30_30/result_EASIS_stent.csv"
	data_folder = "Z:/data/intracranial/data_30_30/stenosis/ESASIS_stent"

	# output_file = "Z:/data/intracranial/data_30_30/result_surgery.csv"
	# data_folder = "Z:/data/intracranial/data_30_30/surgery"

	# create result dataframe
	field_names = ['patient','group','time point',
		'radius mean(mm)','degree of stenosis(%)','radius min(mm)',
		'pressure mean(mmHg)','max pressure gradient(mmHg)','in/out pressure gradient(mmHg)',
		'velocity mean(ms^-1)','peak velocity(ms^-1)',
		'shear strain rate mean(Pas^-1)','peak shear strain rate(Pas^-1)',
		'vorticity mean(s^-1)','peak vorticity(s^-1)']

	result_df = pd.DataFrame(columns=field_names)

	pbar = tqdm(os.listdir(data_folder)[0:])
	# pbar = tqdm(["ChanVaHong"])
	for case in pbar:
		pbar.set_description(case)
		minPoint = (0,0,0)
		maxPoint = (0,0,0)

		if not os.path.exists(os.path.join(data_folder,case)):
			continue

		row = {"patient": case, "group": group, "time point": "baseline"}
		result, mv_matrix_df, mv_dy_matrix_df, minPoint, maxPoint = result_analysis(os.path.join(data_folder,case),minPoint=minPoint,maxPoint=maxPoint,probe=probe)
		row.update(result)
		result_df = result_df.append(pd.Series(row),ignore_index=True)

		# save the moving variance matrix
		mv_matrix_df.to_csv(os.path.join(data_folder,case,"mv_matrix.csv"),index=True)
		mv_dy_matrix_df.to_csv(os.path.join(data_folder,case,"mv_df_matrix.csv"),index=True)

	result_df.to_csv(output_file,index=False)

if __name__=="__main__":
	# main()
	main2()