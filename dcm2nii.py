import SimpleITK as sitk
import os

def readDicom(folder):
	print( "Reading Dicom directory:", folder )
	reader = sitk.ImageSeriesReader()

	dicom_names = reader.GetGDCMSeriesFileNames(folder)
	reader.SetFileNames(dicom_names)

	image = reader.Execute()
	return image

def dcm2nii(srcFolder, tgtFilename):
	# check if srcFolder is empty
	if len(os.listdir(srcFolder)) == 0:
		print("Dicom folder is empty:",srcFolder)
		return

	image = readDicom(srcFolder)
	print( "Writing image:", tgtFilename)
	exit()
	sitk.WriteImage( image, tgtFilename )