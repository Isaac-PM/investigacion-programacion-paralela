#include <iostream>
#include <stdlib.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <vector>
#include "filters/filter.cpp"
#include "compression/compression.cpp"
#include "steganography/multi.cpp"


using namespace std;
using namespace cv;

const string FILTERS_IMAGES_PATH = "images/filters/";
const string FILTERS_IMAGES_PATH_PROCESSED = "images/filters/processed/";
enum class ImageReadResult
{
	SUCCESS,
	FAILURE
};

ImageReadResult readImage(string path, Mat& image)
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

int main(int argc, char** argv)
{
	filter(readImages(FILTERS_IMAGES_PATH + "*.jpg"), FILTERS_IMAGES_PATH, FILTERS_IMAGES_PATH_PROCESSED);
	benchmark(32); //compression
	steganography(20);
	//🙂
	return EXIT_SUCCESS;
}
