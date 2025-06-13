#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdio.h>
#include <helper_cuda.h>
#include <math.h>
#include <iostream>
#include <helper_timer.h>
#include <string>

#include "book.h"
#include "Histogram.h"

#define MAX_THREADS_PER_SM 1536
#define SM_COUNT 14

//int width, hist_width, 
int no_streams = 4;


//kernel for computing histogram right in memory
__global__ void hist_inGlobal (const int* values, int width, int* hist) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	while(idx < width){
		atomicAdd(&hist[values[idx]], 1);
		idx += stride;
	}
}

//computer partial histogram on shared memory and mix them on global memory
__global__ void hist_inShared (const int* values, int width, int* hist){
	extern __shared__ int shHist[];
	shHist[threadIdx.x] = 0;
	__syncthreads();
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	while(idx < width){
		atomicAdd(&shHist[values[idx]], 1);
		idx += stride;
	}
	__syncthreads();
	atomicAdd(&hist[threadIdx.x], shHist[threadIdx.x]);
}

void check_sum(int* hist, int hist_width, int width, int choice) {
	int sum = 0;
    for (int i = 0; i < hist_width; i++)
		sum += hist[i];
	std::string status("FAILED");
	std::string type;
	if (sum == width)
		status = "PASSED";
	if (choice == 2) type = "SERIAL";
	else if(choice == 1) type = "GPU GLOBAL";
	else type = "GPU SHARED";
    std::cout << "(" << type << ")" << "Histogram Check Sum: " << status << '\n';
}


void histogramHost(const int* values, int width, int* hist,
					 int hist_width, int choice){
	int const magic_number = 268435456;
	int *dev_values, *dev_hist;
	int size_dvalues = std::min(width, magic_number) * sizeof(int);
	int size_hist = hist_width * sizeof(int);
	checkCudaErrors(cudaMalloc(&dev_values, size_dvalues));
	checkCudaErrors(cudaMalloc(&dev_hist, size_hist));
	checkCudaErrors(cudaMemcpy(dev_hist, hist, size_hist, cudaMemcpyHostToDevice));
	dim3 blocks(std::min(512, hist_width),1,1);
	dim3 grid(MAX_THREADS_PER_SM/blocks.x*std::floor(SM_COUNT/no_streams), 1, 1); ///);
	std::vector<cudaStream_t> streams(no_streams); 
	for (int i = 0; i < no_streams; i++)
		checkCudaErrors(cudaStreamCreate(&streams[i])); 
	int shMem = blocks.x * sizeof(int);
	int stream_size = size_dvalues / no_streams / sizeof(int);
	for (int i = 0; i < width; i += magic_number) {
		const int* head = values + i;
		if (choice == 0) {
			if (hist_width > 1024) {
				printf("Can't fit histogram inside shared memory.\n");
				break;
			}
			for (int j = 0; j < no_streams; j++)
			{
				checkCudaErrors(cudaMemcpyAsync(
								dev_values + (j * stream_size),
								head + (j * stream_size),
								stream_size * sizeof(int),
								cudaMemcpyHostToDevice, streams[j]
								));
				hist_inShared<<<grid, blocks, shMem, streams[j]>>>
						(dev_values + j * stream_size , stream_size, dev_hist);
			}
		} else if (choice == 1) {
			for (int j = 0; j < no_streams; j++)
			{
				checkCudaErrors(cudaMemcpyAsync(
								dev_values + (j * stream_size),
								head + (j * stream_size),
								stream_size * sizeof(int),
								cudaMemcpyHostToDevice, streams[j]
								));
				hist_inGlobal<<<grid, blocks, 0, streams[j]>>>
						(dev_values + j * stream_size,stream_size, dev_hist);
			}
		}
	}
	cudaDeviceSynchronize();
	cudaCheckLastError(__FILE__, __LINE__);
	for (int i = 0; i < no_streams; i++)
		checkCudaErrors(cudaStreamDestroy(streams[i])); 
	//copy data back to host
	checkCudaErrors(cudaMemcpy(hist,dev_hist,size_hist,cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(dev_hist));
	checkCudaErrors(cudaFree(dev_values));
}

int hist_serial(const int* A, int width, int* hist,
				int hist_width){
	for (int i = 0; i < width; i++)
		hist[A[i]]++;
    return 0;
}

void init_prog (int* A, const int width, int* hist, const int hist_width) {
	srand (time(NULL));
	for (int el = 0; el < width; el++)
		A[el] = rand() % hist_width;
	for (int el = 0; el < hist_width; el++)
		hist[el] = 0;
}

int check_gpu_availability() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        std::cout << "Failed to get device count: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    if (deviceCount == 0) {
        std::cout << "No GPU devices available." << std::endl;
        return 1;
    }
    std::cout << "Found " << deviceCount << " GPU device(s) available." << std::endl;
    return 0;
}

int main(int argc, char* argv[]){
	if (argc < 4) {
		printf("USAGE: width hist_width evaluation [no_streams]\n");
		exit(1);
	}
	check_gpu_availability();
	int iwidth = std::stoi(argv[1]);
	int ihist_width = std::stoi(argv[2]);
	int evaluation = std::stoi(argv[3]);
	if (argc == 5)
		no_streams = std::stoi(argv[4]);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float elapsedTime;
	int serial_time;
	int *A;
	int* hist = new int[ihist_width];
	for (int k = 1; k <= 32; k*=2) {
		printf("ARRAY WIDTH: %d 	HIST WIDTH: %d\n", iwidth * k, ihist_width);
		for (int i = 2; i > -1; i--) {
			A = new int[iwidth*k*sizeof(int)];
			init_prog (A, k * iwidth, hist, ihist_width);
			cudaEventRecord(start, 0);
			if (i == 2) // i chooses the algo
				hist_serial(A, iwidth*k, hist, ihist_width);
			else histogramHost(A, iwidth*k, hist, ihist_width, i);
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&elapsedTime, start, stop);
			if (i == 2) {
				serial_time = static_cast<int>(elapsedTime);
				printf("(SERIAL) elapsed time: %d (ms)\n", serial_time);
			} else if  (i == 1)
				printf("(GLOBAL) elapsed time: %d (ms)	speedup: %f\n", 
							static_cast<int>(elapsedTime), serial_time/elapsedTime);
			else
				printf("(SHARED) elapsed time: %d (ms)  speedup: %f\n", 
							static_cast<int>(elapsedTime), serial_time/elapsedTime);
			if (evaluation == 1)
				check_sum(hist, ihist_width, k * iwidth, i);
 			delete[] A;
		}
		printf("++++++++++++++++++++++++\n");
	}
	delete[] hist;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

//void histogramHost2(const int* values, int width, int* hist,
//					 int hist_width, int choice){
//	//variables
//	int *dev_values, *dev_hist;
//	int size_val = width * sizeof(int);
//	int size_hist = hist_width * sizeof(int);
//	//allocate memory on gpu
//	// checkCudaErrors(cudaMalloc(&dev_values, size_val));
//	checkCudaErrors(cudaMalloc(&dev_hist, size_hist));
//	//copy data to gpu
//	// checkCudaErrors(cudaMemcpy(dev_values, values, size_val, cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMemcpy(dev_hist, hist, size_hist, cudaMemcpyHostToDevice));
//	//configure, lunch and synchronize kernel
//	checkCudaErrors(cudaHostGetDevicePointer((void**) &dev_values, (void*) values, 0));
//	dim3 blocks(hist_width > 256 ? 256 : hist_width,1,1);
//	dim3 grid(ceil(width/((float)blocks.x)),1,1);
//	int shMem = hist_width * sizeof(int);
//	if (choice == 0) {
//		if (hist_width > 12 * 1024 / (1536/blocks.x > 8 ? 8 : 1536/blocks.x)) {
//			printf("Can't fit histogram inside shared memory.\n");
//			return;
//		}
//		hist_inShared<<<grid,blocks,shMem>>>(dev_values, width, dev_hist);
//	} else if (choice == 1) {
//		hist_inGlobal<<<grid,blocks>>>(dev_values, width, dev_hist);
//	}
//	cudaCheckLastError();
//	checkCudaErrors(cudaDeviceSynchronize());
//	//copy data back to host
//	checkCudaErrors(cudaMemcpy(hist,dev_hist,size_hist,cudaMemcpyDeviceToHost));
//	//free GPU memory
//	// checkCudaErrors(cudaFree(dev_values));
//	checkCudaErrors(cudaFree(dev_hist));
//}



//void testHistogram(int choice, const int* A, int width, int* hist, int hist_width){
//	//checkCudaErrors(cudaHostAlloc((void**) &A, width * sizeof(int), cudaHostAllocMapped));
//	//cudaFreeHost(A);
//}
