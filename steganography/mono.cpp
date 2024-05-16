#include <iostream>
#include <stdlib.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <vector>
#include <cstdio>

using namespace std;
using namespace cv;

const string IMAGE_PATH = "containers/mona_lisa.jpg";
const char* INFO_TO_EMBED_PATH = "info/img.jpg";

enum SizeUnit {
    BYTES = 1,
    BITS = 8
};

long getFileSizeIn(SizeUnit unit, FILE* file) {
    if (file == nullptr) return -1;

    long currentPos = ftell(file);
    if (currentPos == -1L) return -1;

    if (fseek(file, 0, SEEK_END) != 0) return -1;

    long size = ftell(file);
    if (size == -1L) return -1;

    if (fseek(file, currentPos, SEEK_SET) != 0) return -1;

    return size * unit;
}

void verifySizeCompatibility(FILE* file, Mat img){
    if (img.channels() != 3) {
        cerr << "Error: Image must be a 3-channel (RGB) image.\n";
        throw runtime_error("Image must be a 3-channel (RGB) image.");
    }

    unsigned int totalBitsToStore = img.rows * img.cols * 3;
    unsigned int fileSizeInBits = getFileSizeIn(SizeUnit::BITS, file);

    if(totalBitsToStore < fileSizeInBits){
        cerr << "Error: A bigger container or smaller info is needed.\n";
        throw runtime_error("A bigger container or smaller info is needed.");
    }
}

void embed(Mat& img, int row, int col, unsigned char bit, int channel) {
    Vec3b& pixel = img.at<Vec3b>(row, col);

    pixel[channel] = (pixel[channel] & ~1) | bit;
}

void readFileAndEmbed(FILE* file, Mat& img) {
    unsigned int fileSize = getFileSizeIn(SizeUnit::BYTES, file);
    int width = img.cols;
    int numChannels = img.channels();

    int row, col;
    unsigned char byte;

    for(unsigned int idx = 0; idx < fileSize; idx++){
        byte = fgetc(file);

        if(byte == EOF) break;

        for(int bitPos = 0; bitPos < 8; bitPos++) {
            unsigned char bit = (byte >> bitPos) & 1;
            int pixelIdx = idx * 8 + bitPos;
            row = pixelIdx / (width * numChannels);
            col = (pixelIdx % (width * numChannels)) / numChannels;
            int channel = (pixelIdx % (width * numChannels)) % numChannels;

            if(row >= img.rows) break;
            embed(img, row, col, bit, channel);
        }
    }

    imwrite("results/mono/result.jpg", img);
}

int main(int argc, char** argv){
    FILE* file = fopen(INFO_TO_EMBED_PATH, "rb");
    Mat img = imread(IMAGE_PATH);

    if(file == nullptr || img.empty()) {
        if (file == nullptr) {
            cerr << "Error: Unable to open the file: " << INFO_TO_EMBED_PATH << endl;
        }
        if (img.empty()) {
            cerr << "Error: Unable to load the image: " << IMAGE_PATH << endl;
        }
        return -1;
    }

    verifySizeCompatibility(file, img);
    readFileAndEmbed(file, img);

    fclose(file);
    return 0;
}
