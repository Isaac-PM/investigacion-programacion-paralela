#include <iostream>
#include <stdlib.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <vector>

using namespace std;
using namespace cv;

// ------------------------------------------------------
// Constants
const string COMPRESSION_IMAGES_PATH = "../images/compression/";
const string COMPRESSION_IMAGES_PATH_PROCESSED = "../images/filters/compression/";
enum class ImageReadResult
{
	SUCCESS,
	FAILURE
};

// ------------------------------------------------------
// Utility functions

ImageReadResult readImage(string path, Mat &image);

vector<Mat> readImages(string path);

// ------------------------------------------------------
// Compression functions



int main(int argc, char **argv)
{
	vector<Mat> images = readImages(COMPRESSION_IMAGES_PATH + "*.tiff");
	for (size_t i = 0; i < images.size(); ++i)
	{
		Mat image = images[i];
		cout << "Image size: " << image.cols << " x " << image.rows << '\n';
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