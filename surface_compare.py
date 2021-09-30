import os
from scipy.sparse import data
import vtk
from tqdm import tqdm
import pandas as pd

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
	data_dir = "Z:/data/intracranial"
	src_dirs = [
		"data_esasis_followup/medical",
		"data_esasis_followup/stent",
		"data_ESASIS_no_stenting",
		"data_wingspan",
		"data_aneurysm_with_stenosis"
	]

	count = 0

	results_df = pd.DataFrame(columns=["case","surface distance","plaque volume", "dos"])

	for src_dir in src_dirs:
		pbar = tqdm(os.listdir(os.path.join(data_dir,src_dir)))

		for case in pbar:
			pbar.set_description(case)

			# load smoothed surface
			smoothed_surface_file = os.path.join(data_dir,src_dir,)

			# load recon surface
			# compute surface distance
			# vtkMassProperty for volume difference (plaque volume)
			

			count += 1
			if count == 10:
				exit()

				
if __name__ == "__main__":
    main()