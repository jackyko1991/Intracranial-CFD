#include <vtkvmtkITKArchetypeImageSeriesScalarReader.h>
#include <vtkvmtkCurvedMPRImageFilter2.h>

#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkMatrix4x4.h>
#include <vtkXMLPolyDataReader.h>
#include <vtkPolyData.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkNIFTIImageWriter.h>

int main(int argc, char *argv[])
{
	std::cout << "Reading image..." << std::endl;
	vtkSmartPointer<vtkvmtkITKArchetypeImageSeriesScalarReader> reader = vtkSmartPointer<vtkvmtkITKArchetypeImageSeriesScalarReader>::New();
	reader->SetArchetype("D:/Dr_Simon_Yu/CFD_intracranial/data/comparison/BlasiRaquelLegaspi/3DRA/3DRA_seg.nii.gz");
	reader->SetDefaultDataSpacing(0.2487, 0.2487, 0.2487);
	reader->SetDefaultDataOrigin(-0, 63.43, -63.43);
	reader->SetOutputScalarTypeToNative();
	reader->SetDesiredCoordinateOrientationToNative();
	reader->SetSingleFile(0);
	reader->Update();

	vtkSmartPointer<vtkImageData> image = vtkSmartPointer<vtkImageData>::New();
	image->DeepCopy(reader->GetOutput());

	vtkSmartPointer<vtkMatrix4x4> matrix = vtkSmartPointer<vtkMatrix4x4>::New();
	matrix->DeepCopy(reader->GetRasToIjkMatrix());

	image->Print(std::cout);
	matrix->Print(std::cout);

	std::cout << "Reading centerline..." << std::endl;

	vtkSmartPointer<vtkXMLPolyDataReader> centerlineReader = vtkSmartPointer<vtkXMLPolyDataReader>::New();
	centerlineReader->SetFileName("D:/Dr_Simon_Yu/CFD_intracranial/data/comparison/BlasiRaquelLegaspi/3DRA/centerline2.vtp");
	centerlineReader->Update();

	vtkSmartPointer<vtkPolyData> centerline = vtkSmartPointer<vtkPolyData>::New();
	centerline->DeepCopy(centerlineReader->GetOutput());
	centerline->Print(std::cout);

	vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
	transform->Scale(-1, 1, 1);

	vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
	transformFilter->SetInputData(centerline);
	//transformFilter->SetInformation(image->GetInformation());
	transformFilter->SetTransform(transform);
	transformFilter->Update();
	centerline->DeepCopy(transformFilter->GetOutput());

	centerline->Print(std::cout);

	std::cout << "Performing MPR..." << std::endl;

	vtkSmartPointer<vtkvmtkCurvedMPRImageFilter2> curvedMPRImageFilter = vtkSmartPointer<vtkvmtkCurvedMPRImageFilter2>::New();
	curvedMPRImageFilter->SetInputData(image);
	curvedMPRImageFilter->SetCenterline(centerline);
	curvedMPRImageFilter->SetParallelTransportNormalsArrayName("ParallelTransportNormals");
	curvedMPRImageFilter->SetFrenetTangentArrayName("FrenetTangent");
	curvedMPRImageFilter->SetInplaneOutputSpacing(0.25, 0.25);
	curvedMPRImageFilter->SetInplaneOutputSize(120, 120);
	curvedMPRImageFilter->SetReslicingBackgroundLevel(-2000);
	curvedMPRImageFilter->Update();

	vtkSmartPointer<vtkImageData> mprImage = vtkSmartPointer<vtkImageData>::New();
	mprImage->DeepCopy(curvedMPRImageFilter->GetOutput());

	std::cout << "Outputing file..." << std::endl;

	vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
	writer->SetFileName("D:/Dr_Simon_Yu/CFD_intracranial/data/comparison/BlasiRaquelLegaspi/3DRA/mpr_seg.vti");
	writer->SetInputData(mprImage);
	writer->Update();

	vtkSmartPointer<vtkNIFTIImageWriter> niiWriter = vtkSmartPointer<vtkNIFTIImageWriter>::New();
	niiWriter->SetFileName("D:/Dr_Simon_Yu/CFD_intracranial/data/comparison/BlasiRaquelLegaspi/3DRA/mpr_seg.nii");
	niiWriter->SetInputData(mprImage);
	niiWriter->Update();
}