#include <fstream>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <string>
#include <tuple>
#include <vector>

using namespace std;
using namespace cv;

// ------------------------------------------------------
// Constants

const string COMPRESSION_IMAGES_PATH = "../images/compression/";
const string COMPRESSION_IMAGES_PATH_PROCESSED = "../images/compression/processed/";
const string BENCHMARK_COLUMNS = "image,resolution,number_of_pixels,compression_rate,number_of_pixel_groups,threads,time";
const string SINGLE_CORE_BENCHMARK_FILE = "single_core.csv";

// ------------------------------------------------------
// Utility functions

enum class ImageReadResult
{
	SUCCESS,
	FAILURE
};

ImageReadResult readImage(string path, Mat &image);

vector<Mat> readImages(string path);

void saveImage(string path, Mat image);

// ------------------------------------------------------
// Compression functions

enum class ImageCompressionRate
{
	LOW = 2,
	MEDIUM = 4,
	HIGH = 8,
	VERY_HIGH = 16
};

const vector<ImageCompressionRate> AllImageCompressionRates = {ImageCompressionRate::LOW, ImageCompressionRate::MEDIUM, ImageCompressionRate::HIGH, ImageCompressionRate::VERY_HIGH};

string parseImageCompressionRate(ImageCompressionRate rate)
{
	switch (rate)
	{
	case ImageCompressionRate::LOW:
		return "low";
	case ImageCompressionRate::MEDIUM:
		return "medium";
	case ImageCompressionRate::HIGH:
		return "high";
	case ImageCompressionRate::VERY_HIGH:
		return "very_high";
	default:
		return "unknown";
	}
}

vector<tuple<unsigned int, unsigned int>> getTopLeftPixelIndexes(unsigned int rows, unsigned int columns, ImageCompressionRate rate)
{
	unsigned int compressionRate = static_cast<unsigned int>(rate);
	vector<tuple<unsigned int, unsigned int>> topLeftPixelIndexes;
	for (size_t i = 0; i < columns; i += compressionRate)
	{
		for (size_t j = 0; j < rows; j += compressionRate)
		{
			topLeftPixelIndexes.push_back(make_tuple(i, j));
		}
	}
	return topLeftPixelIndexes;
}

vector<tuple<unsigned int, unsigned int>> getPixelGroup(tuple<unsigned int, unsigned int> pixelIndex, ImageCompressionRate rate)
{
	unsigned int compressionRate = static_cast<unsigned int>(rate);
	unsigned int x = get<0>(pixelIndex);
	unsigned int y = get<1>(pixelIndex);
	vector<tuple<unsigned int, unsigned int>> adjacentPixelIndexes;
	for (size_t i = 0; i < compressionRate; ++i)
	{
		for (size_t j = 0; j < compressionRate; ++j)
		{
			adjacentPixelIndexes.push_back(make_tuple(x + i, y + j));
		}
	}
	return adjacentPixelIndexes;
}

vector<vector<tuple<unsigned int, unsigned int>>> getAllPixelGroups(Mat image, ImageCompressionRate rate)
{
	vector<vector<tuple<unsigned int, unsigned int>>> pixelGroups;
	vector<tuple<unsigned int, unsigned int>> topLeftPixelIndexes = getTopLeftPixelIndexes(image.rows, image.cols, rate);
	for (size_t i = 0; i < topLeftPixelIndexes.size(); ++i)
	{
		vector<tuple<unsigned int, unsigned int>> adjacentPixelIndexes = getPixelGroup(topLeftPixelIndexes[i], rate);
		pixelGroups.push_back(adjacentPixelIndexes);
	}
	return pixelGroups;
}

Vec3b getAverageFromPixelGroup(Mat image, vector<tuple<unsigned int, unsigned int>> pixelGroup)
{
	vector<unsigned int> redValues, greenValues, blueValues;
	for (size_t i = 0; i < pixelGroup.size(); ++i)
	{
		unsigned int x = get<0>(pixelGroup[i]);
		unsigned int y = get<1>(pixelGroup[i]);
		Vec3b pixel = image.at<Vec3b>(Point(x, y));
		redValues.push_back(pixel[0]);
		greenValues.push_back(pixel[1]);
		blueValues.push_back(pixel[2]);
	}
	unsigned int redAverage = accumulate(redValues.begin(), redValues.end(), 0) / redValues.size();
	unsigned int greenAverage = accumulate(greenValues.begin(), greenValues.end(), 0) / greenValues.size();
	unsigned int blueAverage = accumulate(blueValues.begin(), blueValues.end(), 0) / blueValues.size();
	return Vec3b(redAverage, greenAverage, blueAverage);
}

Mat compressImage(Mat image, ImageCompressionRate rate, size_t &numberOfPixelGroups)
{
	Mat compressedImage;
	unsigned int compressionRate = static_cast<unsigned int>(rate);
	vector<vector<tuple<unsigned int, unsigned int>>> pixelGroups = getAllPixelGroups(image, rate);
	numberOfPixelGroups = pixelGroups.size();
	for (size_t i = 0; i < pixelGroups.size(); ++i)
	{
		Vec3b average = getAverageFromPixelGroup(image, pixelGroups[i]);
		for (size_t j = 0; j < pixelGroups[i].size(); ++j)
		{
			unsigned int x = get<0>(pixelGroups[i][j]);
			unsigned int y = get<1>(pixelGroups[i][j]);
			image.at<Vec3b>(Point(x, y)) = average;
		}
	}
	return image;
}

void singleCoreBenchmark(vector<Mat> images)
{
	fstream file(SINGLE_CORE_BENCHMARK_FILE, ios::out | ios::trunc);
	if (!file.is_open())
	{
		return;
	}

	file << BENCHMARK_COLUMNS << endl;
	for (size_t i = 0; i < images.size(); ++i)
	{
		Mat image = images[i];
		for (ImageCompressionRate rate : AllImageCompressionRates)
		{
			int numberOfPixels = image.cols * image.rows;
			string compressionRate = parseImageCompressionRate(rate);
			size_t numberOfPixelGroups = 0;
			int threads = 1;
			cout << "Processing image " << i << " with resolution " << image.cols << "x" << image.rows << " and compression rate " << compressionRate << " using " << threads << " thread(s)" << endl;
			double startTime = omp_get_wtime();
			Mat compressedImage = compressImage(image.clone(), rate, numberOfPixelGroups);
			double endTime = omp_get_wtime();
			file << i << "," << image.cols << "x" << image.rows << "," << numberOfPixels << "," << compressionRate << "," << numberOfPixelGroups << "," << threads << "," << endTime - startTime << endl;
		}
	}
	file.close();
}

int main(int argc, char **argv)
{
	vector<Mat> images = readImages(COMPRESSION_IMAGES_PATH + "*.tiff");
	singleCoreBenchmark(images);
	return EXIT_SUCCESS;
}

// ------------------------------------------------------
// Utility functions

ImageReadResult readImage(string path, Mat &image)
{
	image = imread(path);
	if (image.empty())
	{
		return ImageReadResult::FAILURE;
	}
	return ImageReadResult::SUCCESS;
}

vector<Mat> readImages(string path)
{
	vector<Mat> images;
	vector<string> filenames;
	glob(path, filenames);
	for (string filename : filenames)
	{
		Mat image;
		if (readImage(filename, image) == ImageReadResult::SUCCESS)
		{
			images.push_back(image);
		}
	}
	return images;
}

void saveImage(string path, Mat image)
{
	imwrite(path, image);
}