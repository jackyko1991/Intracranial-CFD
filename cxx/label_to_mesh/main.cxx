#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkMesh.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkBinaryMask3DMeshSource.h"
#include "itkMeshFileWriter.h"

int
main(int argc, char * argv[])
{
	if (argc != 5)
	{
		std::cerr << "Usage: " << std::endl;
		std::cerr << argv[0];
		std::cerr << " <InputFileName> <OutputFileName> <Lower Threshold> <Upper Threshold>";
		std::cerr << std::endl;
		return EXIT_FAILURE;
	}

	const char * inputFileName = argv[1];
	const char * outputFileName = argv[2];

	constexpr unsigned int Dimension = 3;

	using PixelType = unsigned char;
	using ImageType = itk::Image<PixelType, Dimension>;

	using ReaderType = itk::ImageFileReader<ImageType>;
	ReaderType::Pointer reader = ReaderType::New();
	reader->SetFileName(inputFileName);

	auto lowerThreshold = static_cast<PixelType>(std::stoi(argv[3]));
	auto upperThreshold = static_cast<PixelType>(std::stoi(argv[4]));

	using BinaryThresholdFilterType = itk::BinaryThresholdImageFilter<ImageType, ImageType>;
	BinaryThresholdFilterType::Pointer threshold = BinaryThresholdFilterType::New();
	threshold->SetInput(reader->GetOutput());
	threshold->SetLowerThreshold(lowerThreshold);
	threshold->SetUpperThreshold(upperThreshold);
	threshold->SetOutsideValue(0);

	using MeshType = itk::Mesh<double, Dimension>;

	using FilterType = itk::BinaryMask3DMeshSource<ImageType, MeshType>;
	FilterType::Pointer filter = FilterType::New();
	filter->SetInput(threshold->GetOutput());
	filter->SetObjectValue(255);

	using WriterType = itk::MeshFileWriter<MeshType>;
	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(outputFileName);
	writer->SetInput(filter->GetOutput());
	try
	{
		writer->Update();
	}
	catch (itk::ExceptionObject & error)
	{
		std::cerr << "Error: " << error << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}