#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "kernel.h"

//#define OPENCV_MAT
#define OPENCV_GPUMAT

int main()
{
#ifdef OPENCV_MAT
    // OpenCV CPU version
    Mat raw_input = imread("../Lenna.jpg");
    Mat cv_convert_img;
    Mat cuda_kernel_convert_img;

    //namedWindow("result", WINDOW_NORMAL);
    //setWindowProperty("result", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);

    cv::cvtColor(raw_input, cv_convert_img, COLOR_BGR2RGB);
    imshow("result", cv_convert_img);
    waitKey(0);

    cuda_kernel_convert_img = raw_input.clone();
    ColorBGR2RGB(cuda_kernel_convert_img.data, cuda_kernel_convert_img.rows, cuda_kernel_convert_img.cols, cuda_kernel_convert_img.channels());

    imshow("result", cuda_kernel_convert_img);
    waitKey(0);
    return 0;
#endif

#ifdef OPENCV_GPUMAT
    // OpenCV GPU version
    Mat raw_input = imread("../Lenna.jpg");
    Mat cv_convert_img;
    cv::cuda::GpuMat cuda_kernel_convert_img;
    Mat dst;

    cv::cvtColor(raw_input, cv_convert_img, COLOR_BGR2RGB);
    imshow("result", cv_convert_img);
    waitKey(0);

    cuda_kernel_convert_img.upload(raw_input);
    cv::cuda::GpuMat output(cuda_kernel_convert_img.size(), cuda_kernel_convert_img.type());;

    ColorBGR2RGB_GpuMat(cuda_kernel_convert_img, output, cuda_kernel_convert_img.rows, cuda_kernel_convert_img.cols, cuda_kernel_convert_img.channels());

    output.download(dst);
    imshow("result", dst);
    waitKey(0);
    return 0;
#endif
}
