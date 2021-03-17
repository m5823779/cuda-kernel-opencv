#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.h"
#include <iostream>

__global__ void Conversion_CUDA(unsigned char* d_bgr_image, unsigned char* d_rgb_image, int rows, int cols, int channels)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (col < cols && row < rows)
	{
		int rgb_offset = (row * cols + col) * channels;
		d_rgb_image[rgb_offset + 0] = d_bgr_image[rgb_offset + 2];
		d_rgb_image[rgb_offset + 1] = d_bgr_image[rgb_offset + 1];
		d_rgb_image[rgb_offset + 2] = d_bgr_image[rgb_offset + 0];

	}
}

void ColorBGR2RGB(unsigned char* h_image, int height, int width, int channels) 
{
	unsigned char* d_bgr_image = NULL;
	unsigned char* d_rgb_image = NULL;

	//allocate the memory in gpu
	cudaMalloc((void**)&d_bgr_image, height * width * channels);
	cudaMalloc((void**)&d_rgb_image, height * width * channels);


	//copy data from CPU to GPU
	cudaMemcpy(d_bgr_image, h_image, height * width * channels, cudaMemcpyHostToDevice);

	const dim3 dimGrid((int)ceil((width) / 16), (int)ceil((height) / 16));
	const dim3 dimBlock(16, 16);
	Conversion_CUDA <<<dimGrid, dimBlock>>> (d_bgr_image, d_rgb_image, height, width, channels);

	//copy processed data back to cpu from gpu
	cudaMemcpy(h_image, d_rgb_image, height * width * channels, cudaMemcpyDeviceToHost);

	//free gpu mempry
	cudaFree(d_bgr_image);
	cudaFree(d_rgb_image);
}

void ColorBGR2RGB_GpuMat(unsigned char* h_image, int height, int width, int channels)
{
	unsigned char* d_bgr_image = NULL;
	unsigned char* d_rgb_image = NULL;

	//allocate the memory in gpu
	cudaMalloc((void**)&d_bgr_image, height * width * channels);
	cudaMalloc((void**)&d_rgb_image, height * width * channels);


	//copy data from CPU to GPU
	cudaMemcpy(d_bgr_image, h_image, height * width * channels, cudaMemcpyDeviceToDevice);

	const dim3 dimGrid((int)ceil((width) / 16), (int)ceil((height) / 16));
	const dim3 dimBlock(16, 16);
	Conversion_CUDA << <dimGrid, dimBlock >> > (d_bgr_image, d_rgb_image, height, width, channels);

	//copy processed data back to cpu from gpu
	cudaMemcpy(h_image, d_rgb_image, height * width * channels, cudaMemcpyDeviceToDevice);

	//free gpu mempry
	cudaFree(d_bgr_image);
	cudaFree(d_rgb_image);
}
