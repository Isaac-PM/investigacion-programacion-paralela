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

Mat compressImage(Mat image, ImageCompressionRate rate)
{
	Mat compressedImage;
	unsigned int compressionRate = static_cast<unsigned int>(rate);
	vector<vector<tuple<unsigned int, unsigned int>>> pixelGroups = getAllPixelGroups(image, rate);
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

int main(int argc, char **argv)
{
	vector<Mat> images = readImages(COMPRESSION_IMAGES_PATH + "*.tiff");
	for (size_t i = 0; i < images.size(); ++i)
	{
		Mat image = images[i];
		cout << "Processing image " << i << "..." << endl;
		Mat compressedImageLow = compressImage(image.clone(), ImageCompressionRate::LOW);
		saveImage(COMPRESSION_IMAGES_PATH_PROCESSED + to_string(i) + "_compressed_low.tiff", compressedImageLow);
		Mat compressedImageMedium = compressImage(image.clone(), ImageCompressionRate::MEDIUM);
		saveImage(COMPRESSION_IMAGES_PATH_PROCESSED + to_string(i) + "_compressed_medium.tiff", compressedImageMedium);
		Mat compressedImageHigh = compressImage(image.clone(), ImageCompressionRate::HIGH);
		saveImage(COMPRESSION_IMAGES_PATH_PROCESSED + to_string(i) + "_compressed_high.tiff", compressedImageHigh);
		Mat compressedImageVeryHigh = compressImage(image.clone(), ImageCompressionRate::VERY_HIGH);
		saveImage(COMPRESSION_IMAGES_PATH_PROCESSED + to_string(i) + "_compressed_very_high.tiff", compressedImageVeryHigh);
	}
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