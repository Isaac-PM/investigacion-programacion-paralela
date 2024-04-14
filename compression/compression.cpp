// Author: Isaac Palma Medina @ isaac.palma.medina@est.una.ac.cr

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
const string BENCHMARK_COLUMNS = "image,resolution,number_of_pixels,compression_rate,pixel_group_quantity,pixel_group_quantity_per_thread,threads,time";
const string BENCHMARK_RESULTS = "compression_results.csv";

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

Mat compressImage(Mat image, ImageCompressionRate rate, size_t &pixelGroupQuantity)
{
	Mat compressedImage;
	unsigned int compressionRate = static_cast<unsigned int>(rate);
	vector<vector<tuple<unsigned int, unsigned int>>> pixelGroups = getAllPixelGroups(image, rate);
	pixelGroupQuantity = pixelGroups.size();
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

Mat compressImageThreads(Mat image, ImageCompressionRate rate, size_t &pixelGroupQuantity, size_t &pixelGroupQuantityPerThread, int threads)
{
	Mat compressedImage;
	unsigned int compressionRate = static_cast<unsigned int>(rate);
	vector<vector<tuple<unsigned int, unsigned int>>> pixelGroups = getAllPixelGroups(image, rate);
	pixelGroupQuantity = pixelGroups.size();

	size_t groupsPerThread = pixelGroupQuantity / threads;
	if (groupsPerThread == 0)
	{
		threads = pixelGroupQuantity;
		groupsPerThread = 1;
	}
	pixelGroupQuantityPerThread = groupsPerThread;

#pragma omp parallel num_threads(threads)
	{
		int threadId = omp_get_thread_num();

		size_t startIdx = threadId * groupsPerThread;
		size_t endIdx = (threadId == threads - 1) ? pixelGroupQuantity : startIdx + groupsPerThread;

		for (size_t i = startIdx; i < endIdx; ++i)
		{
			Vec3b average = getAverageFromPixelGroup(image, pixelGroups[i]);
			for (size_t j = 0; j < pixelGroups[i].size(); ++j)
			{
				unsigned int x = get<0>(pixelGroups[i][j]);
				unsigned int y = get<1>(pixelGroups[i][j]);
				image.at<Vec3b>(Point(x, y)) = average;
			}
		}
	}
	return image;
}

void testCompression(string imagePath, string imageName, ImageCompressionRate rate, fstream &file, unsigned int maxThreads)
{
	Mat image = imread(imagePath);
	int numberOfPixels = image.cols * image.rows;
	string compressionRate = parseImageCompressionRate(rate);
	size_t pixelGroupQuantity = 0;
	size_t pixelGroupQuantityPerThread = 0;
	double startTime = 0.0;
	double endTime = 0.0;

	cout << "Processing image " << imagePath << " with resolution " << image.cols << "x" << image.rows << " and compression rate " << compressionRate << " using 1 thread" << endl;
	startTime = omp_get_wtime();
	Mat compressedImage = compressImage(image.clone(), rate, pixelGroupQuantity);
	endTime = omp_get_wtime();
	file << imagePath << "," << image.cols << "x" << image.rows << "," << numberOfPixels << "," << compressionRate << "," << pixelGroupQuantity << "," << pixelGroupQuantity << "," << 1 << "," << endTime - startTime << endl;

	for (unsigned int threads = 2; threads <= maxThreads; threads += 2)
	{
		cout << "Processing image " << imagePath << " with resolution " << image.cols << "x" << image.rows << " and compression rate " << compressionRate << " using " << threads << " threads" << endl;
		startTime = omp_get_wtime();
		Mat compressedImage = compressImageThreads(image.clone(), rate, pixelGroupQuantity, pixelGroupQuantityPerThread, threads);
		endTime = omp_get_wtime();
		saveImage(COMPRESSION_IMAGES_PATH_PROCESSED + imageName + "_" + to_string(threads) + "_" + compressionRate + ".tiff", compressedImage);
		file << imagePath << "," << image.cols << "x" << image.rows << "," << numberOfPixels << "," << compressionRate << "," << pixelGroupQuantity << "," << pixelGroupQuantityPerThread << "," << threads << "," << endTime - startTime << endl;
	}
}

void benchmark(unsigned int maxThreads)
{
	string const LARGE_SIZED_IMAGE = COMPRESSION_IMAGES_PATH + "img_01.tiff";
	string const SMALL_SIZED_IMAGE = COMPRESSION_IMAGES_PATH + "img_05.tiff";
	fstream file(BENCHMARK_RESULTS, ios::out | ios::trunc);
	if (!file.is_open())
	{
		cout << "Error opening file " << BENCHMARK_RESULTS << endl;
		return;
	}
	file << BENCHMARK_COLUMNS << endl;
	
	testCompression(LARGE_SIZED_IMAGE, "img_01", ImageCompressionRate::LOW, file, maxThreads);
	testCompression(LARGE_SIZED_IMAGE, "img_01", ImageCompressionRate::MEDIUM, file, maxThreads);
	testCompression(LARGE_SIZED_IMAGE, "img_01", ImageCompressionRate::HIGH, file, maxThreads);
	testCompression(LARGE_SIZED_IMAGE, "img_01", ImageCompressionRate::VERY_HIGH, file, maxThreads);

	testCompression(SMALL_SIZED_IMAGE, "img_05", ImageCompressionRate::LOW, file, maxThreads);
	testCompression(SMALL_SIZED_IMAGE, "img_05", ImageCompressionRate::MEDIUM, file, maxThreads);
	testCompression(SMALL_SIZED_IMAGE, "img_05", ImageCompressionRate::HIGH, file, maxThreads);
	testCompression(SMALL_SIZED_IMAGE, "img_05", ImageCompressionRate::VERY_HIGH, file, maxThreads);
	file.close();
}

int main(int argc, char **argv)
{
	vector<Mat> images = readImages(COMPRESSION_IMAGES_PATH + "*.tiff");
	benchmark(32);
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