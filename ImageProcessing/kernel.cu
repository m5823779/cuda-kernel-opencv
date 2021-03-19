#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.h"

__global__ void CUDA_ColorConversion(unsigned char* src, unsigned char* dst,
                                     int rows, int cols, int channels) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  if (col < cols && row < rows) {
    int rgb_offset = (row * cols + col) * channels;
    dst[rgb_offset + 0] = src[rgb_offset + 2];
    dst[rgb_offset + 1] = src[rgb_offset + 1];
    dst[rgb_offset + 2] = src[rgb_offset + 0];
  }
}

__global__ void CUDA_ColorConversion(PtrStep<byte> src, PtrStep<byte> dst,
                                     int rows, int cols, int step,
                                     int channels) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  if (col < cols && row < rows) {
    int rgb_offset = (row * step + channels * col);
    dst[rgb_offset + 0] = src[rgb_offset + 2];
    dst[rgb_offset + 1] = src[rgb_offset + 1];
    dst[rgb_offset + 2] = src[rgb_offset + 0];
  }
}

__global__ void CUDA_PixelShifting(PtrStep<byte> src, PtrStep<byte> depth,
                                   PtrStep<byte> dst, int rows, int cols,
                                   int step, int channels) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  if (col < cols && row < rows) {
    int rgb_offset = (row * step + channels * col);
    int depth_offset = (row * depth.step + col);

    int dis = (int)(depth[depth_offset] * 30 / 255);
    if (col > dis) {
      dst[rgb_offset + 0 - dis * channels] = src[rgb_offset + 0];
      dst[rgb_offset + 1 - dis * channels] = src[rgb_offset + 1];
      dst[rgb_offset + 2 - dis * channels] = src[rgb_offset + 2];
    }
  }
}

__global__ void CUDA_ImagePainting(PtrStep<byte> img,
                                   int rows, int cols, int step, int channels) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  if (col < cols && row < rows) {
    int rgb_offset = (row * step + channels * col);
    if (img[rgb_offset + 0] == 0 && img[rgb_offset + 1] == 0 &&
        img[rgb_offset + 2] == 0) {
      for (int offset = 1; offset < 30; offset++) {
        if (col + offset <= cols &&
            img[rgb_offset + 0 + offset * channels] != 0 ||
            img[rgb_offset + 1 + offset * channels] != 0 ||
            img[rgb_offset + 2 + offset * channels] != 0) {
          img[rgb_offset + 0] = img[rgb_offset + 0 + offset * channels];
          img[rgb_offset + 1] = img[rgb_offset + 1 + offset * channels];
          img[rgb_offset + 2] = img[rgb_offset + 2 + offset * channels];
          break;
        }
        if (col + offset <= cols &&
            img[rgb_offset + 0 - offset * channels] != 0 ||
            img[rgb_offset + 1 - offset * channels] != 0 ||
            img[rgb_offset + 2 - offset * channels] != 0) {
          img[rgb_offset + 0] = img[rgb_offset + 0 - offset * channels];
          img[rgb_offset + 1] = img[rgb_offset + 1 - offset * channels];
          img[rgb_offset + 2] = img[rgb_offset + 2 - offset * channels];
          break;
        }
      }
    }
  }
}

void BGR2RGB(unsigned char* h_image, int height, int width, int channels) {
  unsigned char* d_bgr_image = NULL;
  unsigned char* d_rgb_image = NULL;

  // allocate the memory in gpu
  cudaMalloc((void**)&d_bgr_image, height * width * channels);
  cudaMalloc((void**)&d_rgb_image, height * width * channels);

  // copy data from CPU to GPU
  cudaMemcpy(d_bgr_image, h_image, height * width * channels,
             cudaMemcpyHostToDevice);

  const dim3 dimGrid((int)ceil(width / 16.), (int)ceil(height / 16.));
  const dim3 dimBlock(16, 16);
  CUDA_ColorConversion<<<dimGrid, dimBlock>>>(d_bgr_image, d_rgb_image, height,
                                              width, channels);

  // copy processed data back to cpu from gpu
  cudaMemcpy(h_image, d_rgb_image, height * width * channels,
             cudaMemcpyDeviceToHost);

  // free gpu mempry
  cudaFree(d_bgr_image);
  cudaFree(d_rgb_image);
}

void BGR2RGB(PtrStep<byte> src, PtrStep<byte> dst, int height, int width,
             int step, int channels) {
  const dim3 dimGrid((int)ceil(width / 16.), (int)ceil(height / 16.));
  const dim3 dimBlock(16, 16);
  CUDA_ColorConversion<<<dimGrid, dimBlock>>>(src, dst, height, width, step,
                                              channels);
}

void PixelShifting(PtrStep<byte> src, PtrStep<byte> depth, PtrStep<byte> dst,
                   int height, int width, int step, int channels) {
  const dim3 dimGrid((int)ceil(width / 16.), (int)ceil(height / 16.));
  const dim3 dimBlock(16, 16);
  CUDA_PixelShifting<<<dimGrid, dimBlock>>>(src, depth, dst, height, width,
                                            step, channels);
}

void ImagePainting(PtrStep<byte> img, int height, int width,
                   int step, int channels) {
  const dim3 dimGrid((int)ceil(width / 16.), (int)ceil(height / 16.));
  const dim3 dimBlock(16, 16);
  CUDA_ImagePainting<<<dimGrid, dimBlock>>>(img, height, width, step, channels);
}
