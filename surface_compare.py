import os
import vtk
from tqdm import tqdm
import pandas as pd
import json

def surfaceClip(enclose=True):
	return

def volume_similarity_pd(pd1,pd2):
	"""
	computes the following similarity metrics for two polydata objects
	jaccard distance
	hausdorff distance

	returns dict volume_similarity, pd1_with_hausdorff_dist, pd2_with_hausdorff_dist
	"""
	volume_similarity = {}

	# print("aaaaa")

	# union = vtk.vtkBooleanOperationPolyDataFilter()
	# union.SetOperationToDifference()
	# union.SetInputData(0,pd1)
	# union.SetInputData(1,pd2)
	# union.Update()
	# u = union.GetOutput()

	# massUnion = vtk.vtkMassProperties()
	# massUnion.SetInputData(u)

	# intersection = vtk.vtkBooleanOperationPolyDataFilter()
	# intersection.SetOperationToIntersection()
	# intersection.SetInputData(0,pd1)
	# intersection.SetInputData(1,pd2)
	# intersection.Update()
	# i = intersection.GetOutput()
	# massIntersection = vtk.vtkMassProperties()
	# massIntersection.SetInputData(i)

	# # metrics
	# tqdm.write("intersection vol: {:.2f}".format(massIntersection.GetVolume()))
	# tqdm.write("union vol: {:.2f}".format(massUnion.GetVolume()))

	# volume_similarity["jaccard"] = 1 - massIntersection.GetVolume()/massUnion.GetVolume()

	# tqdm.write("Jaccard distance: {:.2f}".format(volume_similarity["jaccard"]))

	hausdorffDistFilter = vtk.vtkHausdorffDistancePointSetFilter()
	hausdorffDistFilter.SetInputData(0, pd1)
	hausdorffDistFilter.SetInputData(1, pd2)
	hausdorffDistFilter.Update()

	volume_similarity["hausdorff"] = hausdorffDistFilter.GetHausdorffDistance()
	volume_similarity["relative0"] = hausdorffDistFilter.GetRelativeDistance()[0]
	volume_similarity["relative1"] = hausdorffDistFilter.GetRelativeDistance()[1]
	tqdm.write("Hausdorff distance: {:.2f} mm".format(volume_similarity["hausdorff"]))
	tqdm.write("Relative distance from pd1 to pd2: {:.2f} mm".format(volume_similarity["relative0"]))
	tqdm.write("Relative distance from pd2 to pd1: {:.2f} mm".format(volume_similarity["relative1"]))

	return volume_similarity, hausdorffDistFilter.GetOutput(0), hausdorffDistFilter.GetOutput(1)

def volume_ratio_pd(pd1,pd2):
	"""

	"""

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

			surface_smoothed_file = os.path.join(data_dir,src_dir,case,"baseline","domain_vessel_smooth.stl")
			surface_recon_file = os.path.join(data_dir,src_dir,case,"baseline","recon_stenosis","source_recon.vtp")
			domain_json = os.path.join(data_dir,src_dir,case,"baseline","domain.json")

			output_file = os.path.join(data_dir,src_dir,case,"baseline","recon_stenosis","surface_distance_smooth.vtp")
			output_file_2 = os.path.join(data_dir,src_dir,case,"baseline","recon_stenosis","surface_distance_smooth_2.vtp")

			# load the files
			if not os.path.exists(surface_smoothed_file) or not os.path.exists(surface_recon_file) or not os.path.exists(domain_json):
				continue

			# read smoothed surfaced
			reader_smooth = vtk.vtkSTLReader()
			reader_smooth.SetFileName(surface_smoothed_file)
			reader_smooth.Update()
			surface_smoothed = reader_smooth.GetOutput()

			reader_recon = vtk.vtkXMLPolyDataReader()
			reader_recon.SetFileName(surface_recon_file)
			reader_recon.Update()
			surface_recon = reader_recon.GetOutput()

			domain = json.load(open(domain_json))

			volume_similarity, surface_recon, surface_smoothed = volume_similarity_pd(surface_recon,surface_smoothed)

			writer = vtk.vtkXMLPolyDataWriter()
			writer.SetFileName(output_file)
			writer.SetInputData(surface_smoothed)
			writer.Update()

			writer.SetFileName(output_file_2)
			writer.SetInputData(surface_recon)
			writer.Update()

			tqdm.write("Output smoothed surface with distance complete")

			# vol1, vol2, delta volume
			mass_smoothed = vtk.vtkMassProperties()
			mass_smoothed.SetInputData(surface_smoothed)
			mass_smoothed.Update()

			mass_recon = vtk.vtkMassProperties()
			mass_recon.SetInputData(surface_recon)
			mass_recon.Update()

			tqdm.write("smoothed surface volume: {:.2f} mm^3".format(mass_smoothed.GetVolume()))
			tqdm.write("recon surface volume: {:.2f} mm^3".format(mass_recon.GetVolume()))
			tqdm.write("stenosis volume: {:.2f} mm^3".format(mass_recon.GetVolume() - mass_smoothed.GetVolume()))

			tqdm.write("smoothed lumenal area: {:.2f} mm^2".format(mass_smoothed.GetSurfaceArea()))
			tqdm.write("recon lumenal area: {:.2f} mm^2".format(mass_recon.GetSurfaceArea()))
			tqdm.write("lumenal area delta: {:.2f} mm^2".format(mass_recon.GetSurfaceArea() - mass_smoothed.GetSurfaceArea()))

			tqdm.write("smoothed normalized shape index: {:.2f}".format(mass_smoothed.GetNormalizedShapeIndex()))
			tqdm.write("recon normalized shape index: {:.2f} mm^2".format(mass_recon.GetNormalizedShapeIndex()))
			tqdm.write("normalized shape index delta: {:.2f} mm^2".format(mass_recon.GetNormalizedShapeIndex() - mass_smoothed.GetNormalizedShapeIndex()))


			count += 1
			if count == 2:
				exit()


if __name__ == "__main__":
    main()