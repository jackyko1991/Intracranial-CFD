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
	data_folder = "I:/CFD/intracranial CBCT 3DRA/comparison/ChanSP/"

	SurfaceDistance(data_folder + '3DRA/surface.stl',data_folder + 'CBCT/surface.stl',data_folder + '/surfaceDistance.vtp')

if __name__ == "__main__":
    main()