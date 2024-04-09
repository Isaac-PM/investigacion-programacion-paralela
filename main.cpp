#include <iostream>
#include <stdlib.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

const string NEGATIVE_IMAGES_PATH = "images/negatives/";

enum class ImageReadResult
{
    SUCCESS,
    FAILURE
};

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

int main(int argc, char **argv)
{
    vector<Mat> images = readImages(NEGATIVE_IMAGES_PATH + "*.jpg");
    for (Mat image : images)
    {
        cout << "Image size: " << image.cols << " x " << image.rows << '\n';
    }
    return EXIT_SUCCESS;
}