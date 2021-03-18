#include <cuda.h>
#include <iostream>
#include <Windows.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace cv::cuda;

void ColorBGR2RGB(unsigned char* h_image, int height, int width, int channels);
void ColorBGR2RGB_GpuMat(PtrStep<byte> src, PtrStep<byte> dst, int height, int width, int channels);





