#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "kernel.h"

//#define OPENCV_MAT
//#define OPENCV_GPUMAT

int main()
{
     Mat raw_input = imread("../Lenna.jpg");

#ifdef OPENCV_MAT
    // OpenCV CPU version
    Mat h_image;

    h_image = raw_input.clone();
    BGR2RGB(h_image.data, h_image.rows, h_image.cols, h_image.channels());

    imshow("Mat result", h_image);
    waitKey(0);
    return 0;
#endif

#ifdef OPENCV_GPUMAT
    // OpenCV GPU version
    Mat dst;

    cv::cuda::GpuMat input;
    input.upload(raw_input);
    cv::cuda::GpuMat output(input.size(), input.type());

    BGR2RGB(input, output, input.rows, input.cols, input.step, input.channels());

    output.download(dst);
    imshow("GpuMat result", dst);
    waitKey(0);
    return 0;
#endif

    Mat dst;
    cv::cuda::GpuMat depth_map(raw_input.rows, raw_input.cols, CV_8UC1,
                               cv::Scalar(10));
    cv::cuda::GpuMat input;
    input.upload(raw_input);
    cv::cuda::GpuMat output(input.size(), input.type());

    PixelShifting(input, depth_map, output, input.rows, input.cols, input.step,
                  input.channels());

    output.download(dst);
    cv::imshow("Pixel shifting", dst);
    cv::waitKey(0);
    return 0;
}
