#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace cv::cuda;

void ColorBGR2RGB(unsigned char* h_image, int height, int width, int channels);
void ColorBGR2RGB_GpuMat(unsigned char* h_image, int height, int width, int channels);





