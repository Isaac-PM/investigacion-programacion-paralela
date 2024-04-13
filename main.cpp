#include <iostream>
#include <stdlib.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <vector>

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
/*
La función applyContrast() recibe una imagen de entrada, un valor de contraste y las dimensiones de la imagen de salida.
#pragma omp parallel for: Esta directiva indica al compilador que debe aplicar el paralelismo en el bucle for externo que itera sobre las filas de la imagen de salida.
Cada hilo creado por OpenMP se encarga de un rango diferente de filas, dividiendo así la carga de trabajo entre múltiples hilos y aprovechando la capacidad de procesamiento multicore.
Dentro de este bucle paralelo, se realiza el procesamiento de píxeles de forma similar a la versión no paralelizada. Cada hilo manipula un conjunto de píxeles independientes,
lo que acelera significativamente el procesamiento en comparación con un enfoque secuencial.
El resultado es una imagen con contraste aplicado, que se devuelve como salida de la función.
*/

Mat applyContrast(const Mat& inputImage, double contrastValue, int targetWidth, int targetHeight)
{
	Mat outputImage;
	resize(inputImage, outputImage, Size(targetWidth, targetHeight));// Redimensionar la imagen de entrada a las dimensiones de salida especificadas

#pragma omp parallel for // Aplicar paralelismo en el bucle for externo que itera sobre las filas de la imagen de salida
	for (int y = 0; y < outputImage.rows; ++y)// Iterar sobre las filas de la imagen de salida
	{
		for (int x = 0; x < outputImage.cols; ++x)// Iterar sobre las columnas de la imagen de salida
		{
			for (int c = 0; c < outputImage.channels(); ++c)// Iterar sobre los canales de color (BGR)
			{
				outputImage.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(
					contrastValue * (outputImage.at<Vec3b>(y, x)[c] - 128) + 128// Aplicar el valor de contraste a cada píxel, utilizando la fórmula de ajuste de contraste
				);
			}
		}
	}

	return outputImage;
}

int main(int argc, char** argv)
{
	cout << "Probando la lectura de imágenes\n";
	vector<Mat> images = readImages(FILTERS_IMAGES_PATH + "*.jpg");
	for (size_t i = 0; i < images.size(); ++i)
	{
		Mat image = images[i];
		cout << "Image size: " << image.cols << " x " << image.rows << '\n';

		// Aplicar contraste
		Mat contrastImage = applyContrast(image, 410.0, 350, 300);  // Puedes ajustar el valor de contraste aquí

		// Mostrar la imagen con contraste
		imshow("Contrast Image", contrastImage);
		waitKey(0);

		// Guardar la imagen con contraste aplicado usando nombres únicos
		string outputFileName = FILTERS_IMAGES_PATH_PROCESSED + "contrast_image_" + to_string(i) + ".jpg";
		imwrite(outputFileName, contrastImage);
	}

	cout << '\n';

	cout << "Probando OpenMP\n";
#pragma omp parallel default(none)
	{
		printf("Thread = %d\n", omp_get_thread_num());
	}

	return EXIT_SUCCESS;
}
