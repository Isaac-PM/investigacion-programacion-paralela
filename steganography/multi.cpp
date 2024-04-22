#include <iostream>
#include <stdlib.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <vector>
#include <cstdio>

//g++ -fopenmp -I/usr/include/opencv4 -L/usr/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui multi.cpp -o multi && ./multi

using namespace std;
using namespace cv;

const string IMAGE_PATH = "containers/mona_lisa.jpg";
const char* INFO_TO_EMBED_PATH = "info/img_prueba.jpeg";
unsigned char* mainBuffer;

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

    cout << img.rows << "x" << img.cols << endl;
    cout << "Total bits to store: " << totalBitsToStore << endl;
    cout << "File size in bits: " << fileSizeInBits << endl;

    if(totalBitsToStore < fileSizeInBits){
        cerr << "Error: A bigger container or smaller info is needed.\n";
        throw runtime_error("A bigger container or smaller info is needed.");
    }
}

void parallelRead(FILE* file, int& fileSize, int numThreads){
    omp_set_num_threads(numThreads);

    fileSize = getFileSizeIn(SizeUnit::BYTES, file);
    mainBuffer = new unsigned char[fileSize];
    int bytesForEachThread = fileSize / numThreads;
    int bytesForLastThread = fileSize % numThreads;

    int start, end;

    #pragma omp parallel for
    for (int i = 0; i < numThreads; i++) {
        int thread = omp_get_thread_num();
        start = thread * bytesForEachThread;
        end = start + bytesForEachThread;

        if (thread == numThreads - 1) {
            end += bytesForLastThread;
        }

        unsigned char* buffer = new unsigned char[end - start];

        fseek(file, start, SEEK_SET);
        fread(buffer, 1, end - start, file);

        #pragma omp critical
        {
            for (int j = start; j < end; j++) {
                mainBuffer[j] = buffer[j - start];
            }
        }

        delete[] buffer;
    }
}

void embed(Mat img, int& fileSize){
    int bitIndex = 0;
    for (int row = 0; row < img.rows; ++row) {
        for (int col = 0; col < img.cols; ++col) {
            Vec3b& pixel = img.at<Vec3b>(row, col);
            for (int channel = 0; channel < 3; ++channel) {
                if (bitIndex / 8 < fileSize) {
                    int bit = (mainBuffer[bitIndex / 8] >> (bitIndex % 8)) & 1;
                    pixel[channel] = (pixel[channel] & ~1) | bit;
                    ++bitIndex;
                }
            }
        }
    }
    imwrite("results/new_img.jpg", img);
}

int main(int argc, char** argv){
    FILE* file = fopen(INFO_TO_EMBED_PATH, "rb");
    int fileSize;

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
    // parallelRead(file, fileSize, 5);
    // embed(img, fileSize);


    fclose(file);
    delete[] mainBuffer;
    return 0;
}
