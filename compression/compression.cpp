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

vector<tuple<int, int>> getTopLeftIndexes(int rows, int columns, ImageCompressionRate rate)
{
	int compressionRate = static_cast<int>(rate);
	vector<tuple<int, int>> topLeftIndexes;
	for (size_t i = 0; i < columns; i += compressionRate)
	{
		for (size_t j = 0; j < rows; j += compressionRate)
		{
			topLeftIndexes.push_back(make_tuple(i, j));
		}
	}
	return topLeftIndexes;
}

int main(int argc, char **argv)
{
	vector<Mat> images = readImages(COMPRESSION_IMAGES_PATH + "*.tiff");
	for (size_t i = 0; i < images.size(); ++i)
	{
		Mat image = images[i];
		cout << "Image size: " << image.cols << " x " << image.rows << '\n';
		vector<tuple<int, int>> topLeftIndexesLow = getTopLeftIndexes(image.rows, image.cols, ImageCompressionRate::VERY_HIGH);
		for (size_t j = 0; j < topLeftIndexesLow.size(); j++)
		{
			int x = get<0>(topLeftIndexesLow[j]);
			int y = get<1>(topLeftIndexesLow[j]);
			cout << "Top left index: " << x << ", " << y << '\n';
		}
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