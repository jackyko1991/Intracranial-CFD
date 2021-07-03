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
from scipy.optimize import curve_fit
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

def fit_func(x, a, b, c):
	return a * b*(x+c) * np.exp(-b * (x+c)**2)

def plot_centerline_result(centerline, array_names, result_path, dev_result_path ,minPoint=(0,0,0),bifurcationPoint=(0,0,0)):
	thresholdFilter = vtk.vtkThreshold()
	thresholdFilter.ThresholdBetween(1,9999)
	thresholdFilter.SetInputData(centerline)
	thresholdFilter.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "CenterlineIds_average")
	thresholdFilter.Update()

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

	fit_dict= {}

	for i in range(len(array_names)):
		popt_list = []

		for lineId in range(int(centerline.GetCellData().GetArray("CenterlineIds_average").GetMaxNorm())):
			thresholdFilter.ThresholdBetween(lineId,lineId)
			thresholdFilter.Update()

			x = vtk_to_numpy(thresholdFilter.GetOutput().GetPointData().GetArray("Abscissas_average"))
			x = [(value - x[0]) for value in x]

			if len(x) < 3:
				continue

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

			# curve fitting on specific values
			if array_names[i] == "Radius_average" or array_names[i] == "U_average" or array_names[i] == "p(mmHg)_average":
				try:
					popt, pcov = curve_fit(fit_func, xs, dys, p0=[50,0.1,-50])

					yfit = fit_func(xs, *popt)
					ax2.plot(xs, fit_func(xs, *popt), 'k-.',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
				except RuntimeError:
					popt = [0,0,0]

				popt_list.append(popt)

		if array_names[i] == "Radius_average":
			ylabel = "Radius (mm)"
			ymin=0
			ymax = 5
			ymin2=-5
			ymax2 = 5
		elif array_names[i] == "U_average":
			ylabel = "Velocity (ms^-1)"
			ymin=0
			ymax = 3
			ymin2=-3
			ymax2 = 3
		elif array_names[i] == "p(mmHg)_average":
			ylabel = "Pressure (mmHg)"
			ymin=0
			ymax = 180
			ymin2=-180
			ymax2 = 180
		elif array_names[i] == "vorticity_average":
			ylabel = "Vorticity (s^-1)"
			ymin=0
			ymax = 4000
			ymin2=-4000
			ymax2 = 4000
		elif array_names[i] == "Curvature_average":
			ylabel = "Curvature"
			ymin = -1.5
			ymax = 1.5
			ymin2 = -1.5
			ymax2 = 1.5
		elif array_names[i] == "Torsion_average":
			ylabel = "Torsion"
			ymin = -100000
			ymax = 100000
			ymin2 = -100000
			ymax2 = 100000

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

		x = vtk_to_numpy(centerline.GetPointData().GetArray("Abscissas_average"))
		ax.set_xlim(x[0],x[-1])
		ax.set_ylim(ymin,ymax)
		ax2.set_xlim(x[0],x[-1])
		ax2.set_ylim(ymin2,ymax2)

		# max of popt[0]
		if len(popt_list) > 0:
			fit_dict.update({array_names[i]: np.abs(np.amax(np.array(popt_list),axis=0)[0])})
		
	# save the plot 
	fig.savefig(result_path,dpi=100)
	fig.clf()

	fig2.savefig(dev_result_path,dpi=100)
	fig2.clf()

	plt.close("all")

	return fit_dict

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

		# transformFilter = vtk.vtkTransformPolyDataFilter()
		transformFilter = vtk.vtkTransformFilter()
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
	fit_dict = plot_centerline_result(
		averageFilter.GetOutput(),
		["U_average","p(mmHg)_average"], 
		#["Radius_average","U_average","p(mmHg)_average","vorticity_average","Curvature_average","Torsion_average"], 
		plot_result_path,
		dev_plot_result_path,
		minPoint = minPoint,
		bifurcationPoint = bifurcationPoint)

	# extract ica
	thresholdFilter = vtk.vtkThreshold()
	thresholdFilter.ThresholdBetween(0,999)
	thresholdFilter.SetInputData(averageFilter.GetOutput())
	thresholdFilter.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "CenterlineIds_average")
	thresholdFilter.Update()

	if thresholdFilter.GetOutput().GetNumberOfPoints() == 0 or fit_dict == None:
		tqdm.write("Centerline file {} does not contain suitable number of CenterlineIds".format(centerline_file))
		return_value =  {
			'radius mean(mm)': "NA",
			'max radius gradient':"NA",
			'radius min(mm)': "NA",
			'pressure mean(mmHg)': "NA",
			'max pressure gradient(mmHg)': "NA",
			'in/out pressure gradient(mmHg)': "NA",
			'velocity mean(ms^-1)': "NA",
			'max velocity gradient(ms^-1)': "NA",
			'peak velocity(ms^-1)': "NA",
			'vorticity mean(s^-1)': "NA",
			'peak vorticity(s^-1)': "NA"
			}
	else:
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
			'max radius gradient': fit_dict["Radius_average"],
			'pressure mean(mmHg)': np.mean(pressure),
			# 'max pressure gradient(mmHg)': np.mean(heapq.nlargest(5, pressure_gradient)),
			'max pressure gradient(mmHg)': fit_dict["p(mmHg)_average"], 
			'in/out pressure gradient(mmHg)': np.mean(pressure[0:5]) - np.mean(pressure[-5:]),
			'velocity mean(ms^-1)': np.mean(velocity),
			'peak velocity(ms^-1)': np.mean(heapq.nlargest(5, velocity)),
			'max velocity gradient(ms^-1)': fit_dict["U_average"],
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

def probe_min_max_point(centerline_filename, surface_filename, minPoint=(0,0,0), maxPoint=(0,0,0),probe=True):
	centerlineReader = vtk.vtkXMLPolyDataReader()
	centerlineReader.SetFileName(centerline_filename)
	centerlineReader.Update()
	centerline = centerlineReader.GetOutput()
	centerline.GetCellData().SetScalars(centerline.GetCellData().GetArray(2));
	centerline.GetPointData().SetScalars(centerline.GetPointData().GetArray("Radius"));

	surfaceReader = vtk.vtkSTLReader()
	surfaceReader.SetFileName(surface_filename)
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
	scalarBar.SetTitle("Radius");
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

	if probe:
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

	tqdm.write("min radius = {:.2f}\nmax radius = {:.2f} mm\nDegree of stenosis = {:.2f} %".format(minRadius,maxRadius,DoS))

	return DoS, minPoint, maxPoint

def translesional_result(centerline_file, wall_file,vtk_file_list, output_dir,minPoint=(0,0,0), prox_dist=5, dist_dist=5):
	# read centerline
	centerlineReader = vtk.vtkXMLPolyDataReader()
	centerlineReader.SetFileName(centerline_file)
	centerlineReader.Update()
	centerline = centerlineReader.GetOutput()

	# read vessel wall
	wallReader = vtk.vtkSTLReader()
	wallReader.SetFileName(wall_file)
	wallReader.Update()

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

		converter2 = vtk.vtkArrayCalculator()
		converter2.SetInputData(converter.GetOutput())
		converter2.AddVectorArrayName("wallShearStress")
		converter2.SetFunction("wallShearStress * 921") # 921 = mu/nu = density of blood, 0.0075 converts from Pascal to mmHg, http://aboutcfd.blogspot.com/2017/05/wallshearstress-in-openfoam.html
		converter2.SetResultArrayName("wallShearStress(Pa)")
		converter2.Update()

		# output the probe centerline
		centerline_output_path = os.path.join(
			os.path.dirname(centerline_file),
			output_dir,
			"centerlines",
			"centerline_probe_{}.vtp".format(os.path.split(file_name)[1].split("_")[1].split(".")[0]) 
			)

		centerlines.append(converter2.GetOutput())
		averageFilter.SetInputData(converter2.GetOutput())
	averageFilter.Update()

	centerline = averageFilter.GetOutput()

	# extract lesion section
	# get the abscissas of min radius point
	# Create kd tree
	kDTree = vtk.vtkKdTreePointLocator()
	kDTree.SetDataSet(centerline)
	kDTree.BuildLocator()

	minIdx = kDTree.FindClosestPoint(minPoint)
	minPoint_absc = centerline.GetPointData().GetArray("Abscissas_average").GetTuple(minIdx)[0]
	thresholdFilter = vtk.vtkThreshold()
	thresholdFilter.ThresholdBetween(minPoint_absc-prox_dist,minPoint_absc+dist_dist)
	thresholdFilter.SetInputData(centerline)
	thresholdFilter.SetAllScalars(0) # important !!!
	thresholdFilter.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,"Abscissas_average");
	thresholdFilter.Update()

	# to vtkpolydata
	geometryFilter = vtk.vtkGeometryFilter()
	geometryFilter.SetInputData(thresholdFilter.GetOutput())
	geometryFilter.Update()

	# get closest line
	connectFilter = vtk.vtkConnectivityFilter()
	connectFilter.SetExtractionModeToClosestPointRegion()
	connectFilter.SetClosestPoint(minPoint)
	connectFilter.SetInputData(geometryFilter.GetOutput())
	connectFilter.Update()

	# compute result values
	abscissas = vtk_to_numpy(thresholdFilter.GetOutput().GetPointData().GetArray("Abscissas_average"))
	abscissas_unique, index = np.unique(sorted(abscissas), axis=0, return_index=True)

	pressure = vtk_to_numpy(thresholdFilter.GetOutput().GetPointData().GetArray("p(mmHg)_average"))
	pressure = [x for _, x in sorted(zip(abscissas,pressure))]
	pressure = [pressure[x] for x in index]
	pressure_gradient = np.diff(pressure)/np.diff(abscissas_unique)

	velocity = vtk_to_numpy(thresholdFilter.GetOutput().GetPointData().GetArray("U_average"))
	velocity = [math.sqrt(value[0]**2 + value[1]**2 +value[2]**2 ) for value in velocity]
	velocity = [x for _, x in sorted(zip(abscissas,velocity))]
	velocity = [velocity[x] for x in index]
	velocity_gradient = np.diff(velocity)/np.diff(abscissas_unique)

	vorticity = vtk_to_numpy(thresholdFilter.GetOutput().GetPointData().GetArray("vorticity_average"))
	vorticity = [math.sqrt(value[0]**2 + value[1]**2 +value[2]**2 ) for value in vorticity]
	vorticity = [x for _, x in sorted(zip(abscissas,vorticity))]
	vorticity = [vorticity[x] for x in index]
	vorticity_gradient = np.diff(vorticity)/np.diff(abscissas_unique)

	wss = vtk_to_numpy(thresholdFilter.GetOutput().GetPointData().GetArray("wallShearStress(Pa)_average"))
	wss = [math.sqrt(value[0]**2 + value[1]**2 +value[2]**2 ) for value in wss]
	wss = [wss[x] for x in index]
	wss = [x for _, x in sorted(zip(abscissas,wss))]
	wss_gradient = np.diff(wss)/np.diff(abscissas_unique)

	epsilon = 0.00001

	p_ratio = np.mean(pressure[-4:-1])/(np.mean(pressure[0:3])+epsilon)
	u_ratio = np.mean(velocity[-4:-1])/(np.mean(velocity[0:3])+epsilon)
	w_ratio = np.mean(vorticity[-4:-1])/(np.mean(vorticity[0:3])+epsilon)
	wss_ratio = np.mean(wss[-4:-1])/(np.mean(wss[0:3])+epsilon)

	dp_ratio = np.mean(pressure_gradient[-4:-1])/(np.mean(pressure_gradient[0:3])+epsilon)
	du_ratio = np.mean(velocity_gradient[-4:-1])/(np.mean(velocity_gradient[0:3])+epsilon)
	dw_ratio = np.mean(vorticity_gradient[-4:-1])/(np.mean(vorticity_gradient[0:3])+epsilon)
	dwss_ratio = np.mean(wss_gradient[-4:-1])/(np.mean(wss_gradient[0:3])+epsilon)

	return_value = {
		'translesion peak presssure(mmHg)': np.mean(heapq.nlargest(1, pressure)),
		'translesion presssure ratio': p_ratio,
		'translesion peak pressure gradient(mmHgmm^-1)': np.mean(heapq.nlargest(1, pressure_gradient)),
		'translesion pressure gradient ratio': dp_ratio,
		'translesion peak velocity(ms^-1)': np.mean(heapq.nlargest(1, velocity)),
		'translesion velocity ratio': u_ratio,
		'translesion velocity gradient ratio': du_ratio,
		'translesion peak velocity gradient(ms^-1mm^-1)': np.mean(heapq.nlargest(1, velocity_gradient)),
		'translesion peak vorticity(ms^-1)': np.mean(heapq.nlargest(1, vorticity)),
		'translesion vorticity ratio': w_ratio,
		'translesion vorticity gradient ratio': dw_ratio,
		'translesion peak vorticity gradient(Pamm^-1)':np.mean(heapq.nlargest(1, vorticity_gradient)),
		'translesion peak wss(Pa)': np.mean(heapq.nlargest(1, wss)),
		'translesion peak wss gradient(Pamm^-1)': np.mean(heapq.nlargest(1, wss_gradient)),
		'translesion wss ratio': wss_ratio,
		'translesion wss gradient ratio': dwss_ratio,
	}

	return return_value

def result_analysis(case_dir, minPoint=(0,0,0), maxPoint=(0,0,0), probe=False ,stenosis=True, perform_fit=False):
	# load domain json file
	with open(os.path.join(case_dir,"domain.json")) as f:
		domain = json.load(f)

	centerline = os.path.join(case_dir, domain["centerline"]["filename"])
	surface = os.path.join(case_dir, domain["domain"]["filename"])
	output_dir = os.path.join(case_dir,"CFD_OpenFOAM_result")

	return_value = {}

	if stenosis:
		try:
			assert domain["fiducial_0"]["type"] == "Stenosis", "\"fiducial_0\" is not Stenosis"
			minPoint = tuple(domain["fiducial_0"]["coordinate"])
		except:
			minPoint = (0,0,0)

		try:
			assert domain["fiducial_1"]["type"] == "DoS_Ref", "\"fiducial_1\" is not DoS_Ref"
			maxPoint = tuple(domain["fiducial_1"]["coordinate"])
		except:
			maxPoint = (0,0,0)


		DoS, minPoint, maxPoint = probe_min_max_point(centerline,surface,minPoint,maxPoint, probe=probe)

		if minPoint != (0,0,0):
			domain["fiducial_0"] = {"coordinate": minPoint, "type": "Stenosis"}
			# translesional values
			# load last 5 time points and take average
			results_vtk = []

			for time in range(0,2001,100):
				results_vtk.append(os.path.join(case_dir,"CFD_OpenFOAM", "VTK","OpenFOAM_" + str(time)+".vtk"))
			return_value_ = translesional_result(
				os.path.join(case_dir,domain["centerline"]["filename"]),
				os.path.join(case_dir,domain["vessel"]["filename"]),
				results_vtk[-5:],
				output_dir, 
				minPoint=minPoint
				)
			return_value.update(return_value_)
		if maxPoint != (0,0,0):
			domain["fiducial_1"] = {"coordinate": maxPoint, "type": "DoS_Ref"}

		with open(os.path.join(case_dir,"domain.json"), 'w') as f:
			json.dump(domain, f)		
	else:
		DoS = 0

	return_value['degree of stenosis(%)'] = DoS

	if perform_fit:
		# load last 5 time points and take average
		results_vtk = []

		for time in range(0,2001,100):
			results_vtk.append(os.path.join(case_dir,"CFD_OpenFOAM", "VTK","OpenFOAM_" + str(time)+".vtk"))
		
		return_value_, mv_matrix_df, mv_dy_matrix_df = centerline_probe_result(
			os.path.join(case_dir,domain["centerline"]["filename"]),
			results_vtk[-5:],
			output_dir, 
			minPoint=minPoint,
			bifurcationPoint=domain["bifurcation_point"]["coordinate"]
			)

		return_value.update(return_value_)

	if perform_fit:
		return return_value, mv_matrix_df, mv_dy_matrix_df, minPoint, maxPoint
	else:
		return return_value, minPoint, maxPoint

def main():
	probe=False
	stenosis=True
	perform_fit = True
	use_case_list = False

	# output_file = "Z:/data/intracranial/CFD_results/result_medical.csv"
	# data_folder = "Z:/data/intracranial/data_ESASIS_followup/medical"

	# output_file = "Z:/data/intracranial/CFD_results/result_stent.csv"
	# data_folder = "Z:/data/intracranial/data_ESASIS_followup/stent"

	# output_file = "Z:/data/intracranial/CFD_results/result_data_no_stenting.csv"
	# data_folder = "Z:/data/intracranial/data_ESASIS_no_stenting"

	# output_file = "Z:/data/intracranial/CFD_results/result_data_surgery.csv"
	# data_folder = "Z:/data/intracranial/data_surgery"

	# output_file = "Z:/data/intracranial/CFD_results/result_wingspan.csv"
	# data_folder = "Z:/data/intracranial/data_wingspan"

	#output_file = "Z:/data/intracranial/CFD_results/result_aneurysm_with_stenosis_2.csv"
	#data_folder = "Z:/data/intracranial/data_aneurysm_with_stenosis"

	output_file = "Y:/data/intracranial/CFD_results/result_medical_001.csv"
	data_folder = "Y:/data/intracranial/data_ESASIS_followup/medical"

	# create result dataframe
	field_names = ['patient','stage',
		'radius mean(mm)','degree of stenosis(%)','radius min(mm)',
		'max radius gradient',
		'pressure mean(mmHg)','max pressure gradient(mmHg)','in/out pressure gradient(mmHg)',
		'velocity mean(ms^-1)','peak velocity(ms^-1)','max velocity gradient(ms^-1)',
		'vorticity mean(s^-1)','peak vorticity(s^-1)',
		'translesion peak presssure(mmHg)',
		'translesion presssure ratio',
		'translesion peak pressure gradient(mmHgmm^-1)',
		'translesion pressure gradient ratio',
		'translesion peak velocity(ms^-1)',
		'translesion velocity ratio',
		'translesion peak velocity gradient(ms^-1mm^-1)',
		'translesion velocity gradient ratio',
		'translesion peak vorticity(ms^-1)',
		'translesion vorticity ratio',
		'translesion peak vorticity gradient(Pamm^-1)',
		'translesion vorticity gradient ratio',
		'translesion peak wss(Pa)',
		'translesion wss ratio',
		'translesion peak wss gradient(Pamm^-1)',
		'translesion wss gradient ratio',
		]

	result_df = pd.DataFrame(columns=field_names)

	case_list = os.listdir(data_folder)
	case_list = ["001"]
	pbar = tqdm(case_list)

	ignore_case = [

	]

	stages = ["baseline"]

	if use_case_list:
		run_list = "./run_list.csv"
		with open(run_list, newline='') as csvfile:
			reader = csv.reader(csvfile)
			run_cases = [l[0] for l in reader]

	for case in pbar:
		pbar.set_description(case)
		minPoint = (0,0,0)
		maxPoint = (0,0,0)

		working_dir = os.path.join(data_folder,case,"baseline")

		if not os.path.exists(working_dir) or not os.path.exists(os.path.join(working_dir,"domain.json")):
			continue

		if use_case_list:
			if not case in run_cases:
				continue

		if case in ignore_case:
			continue

		for stage in stages:
			row = {"patient": case, "stage": stage}
			if perform_fit:
				result, mv_matrix_df, mv_dy_matrix_df, minPoint, maxPoint = result_analysis(os.path.join(data_folder,case,stage),minPoint=minPoint,maxPoint=maxPoint,probe=probe, perform_fit=perform_fit)
			
				# save the moving variance matrix
				mv_matrix_df.to_csv(os.path.join(data_folder,case,"mv_matrix.csv"),index=True)
				mv_dy_matrix_df.to_csv(os.path.join(data_folder,case,"mv_df_matrix.csv"),index=True)
			else:
				result, minPoint, maxPoint = result_analysis(os.path.join(data_folder,case,stage),minPoint=minPoint,maxPoint=maxPoint,probe=probe, stenosis=stenosis, perform_fit=perform_fit)

			row.update(result)
			result_df = result_df.append(pd.Series(row),ignore_index=True)

	result_df.to_csv(output_file,index=False)

if __name__=="__main__":
	main()