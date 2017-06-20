import os
import vtk

def SurfaceDistance(threeDRA_path,CBCT_path,output_path):
	reader3DRA = vtk.vtkSTLReader()
	reader3DRA.SetFileName(threeDRA_path)
	reader3DRA.Update()
	threeDRA = reader3DRA.GetOutput()

	readerCBCT = vtk.vtkSTLReader()
	readerCBCT.SetFileName(CBCT_path)
	readerCBCT.Update()
	CBCT = readerCBCT.GetOutput()

	distanceFilter = vtk.vtkDistancePolyDataFilter()
	distanceFilter.SetInputData( 0, threeDRA)
  	distanceFilter.SetInputData( 1, CBCT)
  	distanceFilter.Update()

  	# extract lcc object
  	connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
  	connectivityFilter.SetInputData(distanceFilter.GetOutput())
	connectivityFilter.Update()

  	writer = vtk.vtkXMLPolyDataWriter()
  	writer.SetFileName(output_path)
  	writer.SetInputData(connectivityFilter.GetOutput())
  	writer.Update()

def main():
	data_folder = "D:/Cloud/Google Drive/intracranial vessels/followup/stent/"
	for patient in os.listdir(data_folder):
		print "Working on patient",patient

		if not os.path.isfile(data_folder + patient + '/baseline/surface.stl') :
			print patient + '/baseline/surface.stl not exist, case skipped'
			return

		if os.path.isfile(data_folder + patient + '/baseline-post/surface.stl') :
			print "Progress: post stent"
			SurfaceDistance(data_folder + patient +  '/baseline/surface.stl',data_folder + patient + '/baseline-post/surface.stl',data_folder + patient + '/baseline-post/surfaceDistance.vtp')
		else:
			print patient + '/baseline-post/surface.stl not exist'

		if os.path.isfile(data_folder + patient + '/12months/surface.stl') :
			print "Progress: 12 months"
			SurfaceDistance(data_folder + patient +  '/baseline/surface.stl',data_folder + patient + '/12months/surface.stl',data_folder + patient + '/12months/surfaceDistance.vtp')
		else:
			print patient + '/12months/surface.stl not exist'
		
		if os.path.isfile(data_folder + patient + '/followup/surface.stl') :
			print "Progress: followup"
			SurfaceDistance(data_folder + patient +  '/baseline/surface.stl',data_folder + patient + '/followup/surface.stl',data_folder + patient + '/followup/surfaceDistance.vtp')
		else:
			print patient + '/followup/surface.stl not exist'

if __name__ == "__main__":
    main()