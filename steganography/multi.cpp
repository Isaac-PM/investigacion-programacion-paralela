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

struct ByteData;

const string IMAGE_PATH = "containers/mona_lisa.jpg";
const char* INFO_TO_EMBED_PATH = "info/img.jpg";

unsigned char* mainBuffer;
int* threadsStartPositionsBuffer;
int* threadsEndPositionsBuffer;
ByteData* byteDataBuffer;

struct ByteData{
    unsigned char byte;
    size_t posInMainBuffer;
    size_t row;
    size_t col;
};

int getFileSize(FILE* file) {
    fseek(file, 0, SEEK_END);
    int size = ftell(file);
    fseek(file, 0, SEEK_SET);
    return size;
}

void verifySizeCompatibility(FILE* file, Mat img){
    if (img.channels() != 3) {
        cerr << "Error: Image must be a 3-channel (RGB) image.\n";
        throw runtime_error("Image must be a 3-channel (RGB) image.");
    }

    unsigned int totalBitsToStore = img.rows * img.cols * 3;
    totalBitsToStore = totalBitsToStore - (img.rows * img.cols / 3); // restar 1 bit cada 3 pixeles
    unsigned int fileSizeInBits = getFileSize(file);

    if(totalBitsToStore < fileSizeInBits){
        cerr << "Error: A bigger container or smaller info is needed.\n";
        throw runtime_error("A bigger container or smaller info is needed.");
    }
}

void embed(Mat &img, int thread){
    for(size_t i = threadsStartPositionsBuffer[thread]; i < threadsEndPositionsBuffer[thread]; i++){
        ByteData byte = byteDataBuffer[i];
        unsigned char bits[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for(int j = 0; j < 8; j++){
            bits[j] = (byte.byte >> j) & 1;
        }
        
        for(int i = 0; i < 3; i++){
            Vec3b pixel = img.at<Vec3b>(byte.row, byte.col + i);

            if(i == 0){ // si está en el primer pixel
                pixel[0] = (pixel[0] & 0xFE) | bits[0];
                pixel[1] = (pixel[1] & 0xFE) | bits[1];
                pixel[2] = (pixel[2] & 0xFE) | bits[2];
            } else if(i == 1){ // si está en el segundo pixel
                pixel[0] = (pixel[0] & 0xFE) | bits[3];
                pixel[1] = (pixel[1] & 0xFE) | bits[4];
                pixel[2] = (pixel[2] & 0xFE) | bits[5];
            } else { // si está en el tercer pixel
                pixel[0] = (pixel[0] & 0xFE) | bits[6];
                pixel[1] = (pixel[1] & 0xFE) | bits[7];
            }
        }

        /*
        En binario, 0xFE es 11111110. Al realizar una operación AND entre este valor y otro byte, 
        se garantiza que todos los bits del otro byte se conserven, excepto el bit menos significativo (LSB), 
        que se establecerá en 0, para después hacerle un OR y poner el LSB que se desea.
        */
    }
}

void parallelRead(const char* fileName, Mat img, int& fileSize, int numThreads){
    FILE* file = fopen(fileName, "rb");
    if (!file) {
        perror("Failed to open file");
        return;
    }

    fileSize = getFileSize(file);
    fclose(file);

    mainBuffer = new unsigned char[fileSize];
    byteDataBuffer = new ByteData[fileSize];
    threadsStartPositionsBuffer = new int[numThreads];
    threadsEndPositionsBuffer = new int[numThreads];
    int bytesForEachThread = fileSize / numThreads;
    int bytesForLastThread = fileSize % numThreads;

    omp_set_num_threads(numThreads);
    #pragma omp parallel
    {
        int thread = omp_get_thread_num();
        int start = thread * bytesForEachThread;
        int end = (thread == numThreads - 1) ? start + bytesForEachThread + bytesForLastThread : start + bytesForEachThread;

        threadsStartPositionsBuffer[thread] = start;
        threadsEndPositionsBuffer[thread] = end;

        FILE* file = fopen(fileName, "rb");
        if (file) {
            fseek(file, start, SEEK_SET);
            fread(mainBuffer + start, 1, end - start, file);
            fclose(file);
        } else {
            printf("Error en el hilos %d\n", thread);
        }
    }

    size_t row = 0;
    size_t col = 0;
    for(size_t i = 0; i < fileSize; i++){
        if(col % img.cols == 0 && i != 0){
            row++;
        }
        ByteData byte = {mainBuffer[i], i, row, col%img.cols};
        byteDataBuffer[i] = byte;
        col += 3;
    }

    delete[] mainBuffer;
}

void parallelEmbed(Mat &img, int numThreads){
    omp_set_num_threads(numThreads);
    #pragma omp parallel
    {
        int thread = omp_get_thread_num();
        embed(img, thread);
    }

    imwrite("results/multi/result.jpg", img);
}

int steganography(int numThreads){
    const char* fileName = "./info/img.jpg";
    FILE* file = fopen(fileName, "rb");
    Mat img = imread("./containers/mona_lisa.jpg");
    int fileSize;


    verifySizeCompatibility(file, img);
    parallelRead(fileName, img, fileSize, numThreads);
    parallelEmbed(img, numThreads);

    delete[] byteDataBuffer;
    delete[] threadsStartPositionsBuffer;
    delete[] threadsEndPositionsBuffer;

    return 0;
}
