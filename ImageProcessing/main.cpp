#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "kernel.h"

//#define OPENCV_MAT
#define OPENCV_GPUMAT

int main() {
  Mat raw_input = imread("../test.jpg");
  Mat opencv_api_speed_test = raw_input.clone();
  clock_t process_start;
  clock_t process_end;
  double process_duration;
  
  process_start = clock();
  cv::cvtColor(opencv_api_speed_test, opencv_api_speed_test, COLOR_BGR2RGB);
  process_end = clock();

  process_duration = process_end - process_start;
  cout << "OpenCV color space conversion API (BGR2RGB) :  " << process_duration << " (ms)" << endl;

#ifdef OPENCV_MAT
  // OpenCV CPU version
  Mat h_image;
  h_image = raw_input.clone();

  process_start = clock();
  BGR2RGB(h_image.data, h_image.rows, h_image.cols, h_image.channels());
  process_end = clock();

  process_duration = process_end - process_start;
  cout << "OpenCV Mat color space conversion with cuda (BGR2RGB) :  " << process_duration << " (ms)" << endl;

  imshow("Color space conversion", h_image);
  waitKey(0);
#endif

#ifdef OPENCV_GPUMAT
  // OpenCV GPU version
  Mat dst_rgb;
  cv::cuda::GpuMat input_bgr;
  input_bgr.upload(raw_input);
  cv::cuda::GpuMat output_rgb(input_bgr.size(), input_bgr.type());

  process_start = clock();
  BGR2RGB(input_bgr, output_rgb, input_bgr.rows, input_bgr.cols, input_bgr.step,
          input_bgr.channels());
  process_end = clock();
  
  process_duration = process_end - process_start;
  cout << "OpenCV GpuMat color space conversion with cuda (BGR2RGB) :  " << process_duration << " (ms)" << endl;

  output_rgb.download(dst_rgb);
  imshow("Color space conversion", dst_rgb);
  waitKey(0);
#endif

  // pixel shifting
  Mat dst_pixel_shifting;
  cv::cuda::GpuMat depth_map(raw_input.rows, raw_input.cols, CV_32FC1,
                             cv::Scalar(0.5));
  depth_map.convertTo(depth_map, CV_8UC1, 255.0 / 1.0, 0);

  cv::cuda::GpuMat input_pixel_shifting;
  input_pixel_shifting.upload(raw_input);
  cv::cuda::GpuMat output_pixel_shifting(input_pixel_shifting.size(),
                          input_pixel_shifting.type());

  process_start = clock();
  PixelShifting(input_pixel_shifting, depth_map, output_pixel_shifting,
                input_pixel_shifting.rows, input_pixel_shifting.cols,
                input_pixel_shifting.step, input_pixel_shifting.channels());
  process_end = clock();

  process_duration = process_end - process_start;
  cout << "OpenCV GpuMat pixel shifting with cuda :  "
       << process_duration << " (ms)" << endl;

  output_pixel_shifting.download(dst_pixel_shifting);
  cv::imshow("Pixel shifting", dst_pixel_shifting);
  cv::waitKey(0);

  // image inpainting
  Mat input_inpainting = dst_pixel_shifting.clone();
  Mat fixed_image;
  cv::cuda::GpuMat image_inpainting;
  image_inpainting.upload(input_inpainting);
  cv::cuda::GpuMat output_inpainting_gpumat(image_inpainting.size(),
                                            image_inpainting.type());
  
  process_start = clock();
  ImagePainting(image_inpainting, image_inpainting.rows, image_inpainting.cols,
                image_inpainting.step, image_inpainting.channels());
  process_end = clock();

  process_duration = process_end - process_start;
  cout << "OpenCV GpuMat image inpainting with cuda :  " << process_duration
       << " (ms)" << endl;

  image_inpainting.download(fixed_image);
  cv::imshow("Image inpainting", fixed_image);
  cv::waitKey(0);
  return 0;
}
