#include "iostream"
#include "vtkXMLPolyDataReader.h"
#include "vtkXMLPolyDataWriter.h"
#include "vtkSmartPointer.h"
#include "vtkPolyData.h"

#include "vtkvmtkPolyDataFlowExtensionsFilter.h"

int main(int argc, char *argv[])
{
	//if (argc != 4)
	//{
	//	std::cerr << "Usage: " << std::endl;
	//	std::cerr << argv[0];
	//	std::cerr << " <InputSurface> <InputCenterline> <OutputFolder>";
	//	std::cerr << std::endl;
	//	return EXIT_FAILURE;
	//}

	//const char * InputSurfaceFilename = argv[1];
	//const char * InputCenterlineFilename = argv[2];
	//const char * outputFolder = argv[3];

	// read the input surface and centerline
	vtkSmartPointer<vtkXMLPolyDataReader> reader = vtkSmartPointer<vtkXMLPolyDataReader>::New();
	//reader->SetFileName(InputSurfaceFilename);
	reader->SetFileName("D:/Dr_Simon_Yu/CFD_intracranial/data/comparison/BlasiRaquelLegaspi/3DRA/surface_clipped.vtp");
	reader->Update();


	vtkSmartPointer<vtkPolyData> surface = vtkSmartPointer<vtkPolyData>::New();
	surface->DeepCopy(reader->GetOutput());

	//reader->SetFileName(InputCenterlineFilename);
	reader->SetFileName("D:/Dr_Simon_Yu/CFD_intracranial/data/comparison/BlasiRaquelLegaspi/3DRA/centerline_clipped.vtp");
	reader->Update();

	vtkSmartPointer<vtkPolyData> centerline = vtkSmartPointer<vtkPolyData>::New();
	centerline->DeepCopy(reader->GetOutput());

	vtkSmartPointer<vtkvmtkPolyDataFlowExtensionsFilter> flowExtensionFilter = vtkSmartPointer<vtkvmtkPolyDataFlowExtensionsFilter>::New();
	flowExtensionFilter->SetInputData(surface);
	flowExtensionFilter->SetCenterlines(centerline);
	flowExtensionFilter->SetSigma(1.0);
	flowExtensionFilter->SetAdaptiveExtensionLength(1);
	flowExtensionFilter->SetAdaptiveExtensionRadius(1);
	flowExtensionFilter->SetAdaptiveNumberOfBoundaryPoints(1);
	flowExtensionFilter->SetExtensionLength(1.0);
	flowExtensionFilter->SetExtensionRatio(3.0);
	flowExtensionFilter->SetExtensionRadius(1.0);
	flowExtensionFilter->SetTransitionRatio(1.0);
	flowExtensionFilter->SetCenterlineNormalEstimationDistanceRatio(1.0);
	flowExtensionFilter->SetNumberOfBoundaryPoints(1);
	flowExtensionFilter->SetExtensionModeToUseCenterlineDirection();
	flowExtensionFilter->SetInterpolationModeToThinPlateSpline();
	flowExtensionFilter->Update();

	vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
	//writer->SetFileName(outputFolder + "/surface_extended.vtp");
	writer->SetInputData(flowExtensionFilter->GetOutput());
	writer->SetFileName("D:/Dr_Simon_Yu/CFD_intracranial/data/comparison/BlasiRaquelLegaspi/3DRA/surface_extended.vtp");
	writer->Update();

	return EXIT_SUCCESS;
}