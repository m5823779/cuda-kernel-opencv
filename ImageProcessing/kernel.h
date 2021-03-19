#include <Windows.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace cv::cuda;

void BGR2RGB(unsigned char* h_image, int height, int width, int channels);
void BGR2RGB(PtrStep<byte> src, PtrStep<byte> dst, int height, int width,
             int step, int channels);
void PixelShifting(PtrStep<byte> src, PtrStep<byte> depth, PtrStep<byte> dst,
                   int height, int width, int step, int channels);
void ImagePainting(PtrStep<byte> img, int height, int width, int step,
                   int channels);
