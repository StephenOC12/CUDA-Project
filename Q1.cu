//============================================================================
// Name        : Q1.cpp
// Author      : 
// Version     :
// Copyright   :
// Description : COM2039 Histogram Coursework
//============================================================================

#include "com2039.hpp"

/// Error Checking
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/// Loading file
size_t loadSamples(const char* path_to_data_points_file, float** ptr ){
    std::ifstream file (path_to_data_points_file, std::ios::in|std::ios::binary|std::ios::ate);
    std::streampos size_read = file.tellg();
    if (size_read < 0){
        std::cout << "Error reading file " << path_to_data_points_file << std::endl;
        exit(1);
    }
    size_t len_array = size_read/sizeof(float);
    std::cout << "Read :" << size_read << " bytes = " << len_array << " elements." << std::endl;

    char* memblock = new char[size_read];
    file.seekg(0, std::ios::beg);
    file.read (memblock, size_read);    file.close();
    std::cout << "Correctly loaded "<< path_to_data_points_file << std::endl;
    *ptr = (float*)memblock;

    return len_array;
}

/////// Find Maximum
__global__ void maxReduceKernel(float *d_in, size_t len_array){
	__shared__ float sdata[BLOCK_SIZE];

	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < len_array) {
		sdata[tid] = d_in[idx];
	} else {
		sdata[tid] = -FLT_MAX;
	}

	__syncthreads();

	for (unsigned int stride = blockDim.x / 2 ; stride > 0; stride >>= 1) {
		if (tid < stride) {
			sdata[tid] = max(sdata[tid],sdata[tid + stride]);
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		d_in[blockIdx.x] = sdata[0];
	}

}


float findMaxValue(float* samples_h, size_t len_array){
	float* input_d;
	size_t size = len_array * sizeof(float);

	cudaError_t err;
	err = cudaMalloc((void**) &input_d, size);
	if (err != cudaSuccess) {
		std::cout << "Cuda Error allocating memory for input_d: " <<cudaGetErrorString(err) << std::endl;
		exit(-1);
	}

	err = cudaMemcpy(input_d, samples_h, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cout << "Cuda Error with Memcpy: " <<cudaGetErrorString(err) << std::endl;
		cudaFree(input_d);
		exit(-1);
	}

	size_t currentSize = len_array;
	int num_blocks = (currentSize + BLOCK_SIZE -1) / BLOCK_SIZE;

	while(currentSize > 1) {
		maxReduceKernel<<<num_blocks, BLOCK_SIZE>>>(input_d, currentSize);
		cudaDeviceSynchronize();
		currentSize = num_blocks;
		num_blocks = (currentSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
	}

	cudaDeviceSynchronize();

	float result;
	err = cudaMemcpy(&result, input_d, sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cout << "Cuda Error with Memcpy: " <<cudaGetErrorString(err) << std::endl;
		cudaFree(input_d);
		exit(-1);
	}

	cudaFree(input_d);
	return result;
}


/////// Find Minimum
__global__ void minReduceKernel(float *d_in, size_t len_array){
	__shared__ float sdata[BLOCK_SIZE];

	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < len_array) {
		sdata[tid] = d_in[idx];
	} else {
		sdata[tid] = FLT_MAX;
	}

	__syncthreads();

	for (int stride = blockDim.x / 2 ; stride > 0; stride >>= 1) {
		if (tid < stride) {
			sdata[tid] = min(sdata[tid] , sdata[tid + stride]);
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		d_in[blockIdx.x] = sdata[0];
	}

}


float findMinValue(float* samples_h, size_t len_array){
	float* input_d;
	size_t size = len_array * sizeof(float);

	cudaError_t err;
	err = cudaMalloc((void**) &input_d, size);
	if (err != cudaSuccess) {
		std::cout << "Cuda Error allocating memory for input_d: " <<cudaGetErrorString(err) << std::endl;
		exit(-1);
	}

	err = cudaMemcpy(input_d, samples_h, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cout << "Cuda Error with Memcpy: " <<cudaGetErrorString(err) << std::endl;
		cudaFree(input_d);
		exit(-1);
	}

	size_t currentSize = len_array;
	int num_blocks = (currentSize + BLOCK_SIZE -1) / BLOCK_SIZE;

	while(currentSize > 1) {
		minReduceKernel<<<num_blocks, BLOCK_SIZE>>>(input_d, currentSize);
		cudaDeviceSynchronize();
		currentSize = num_blocks;
		num_blocks = (currentSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
	}

	cudaDeviceSynchronize();

	float result;
	err = cudaMemcpy(&result, input_d, sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cout << "Cuda Error with Memcpy: " <<cudaGetErrorString(err) << std::endl;
		cudaFree(input_d);
		exit(-1);
	}

	cudaFree(input_d);
	return result;
}



/////// Create Histogram
__global__ void histogramKernel512(float *d_in, unsigned int *hist, size_t len_array, float min_value, float max_value) {
	__shared__ unsigned int hist_shared[NUM_BINS];

	int tid = threadIdx.x;
	if (tid < NUM_BINS) {
		hist_shared[tid] = 0;
	}
	__syncthreads();

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < len_array; i += stride) {
		float val = d_in[i];
		int bin = (int)(((val-min_value) / (max_value-min_value)) * NUM_BINS);
		bin = min(max(bin, 0), static_cast<int>(NUM_BINS - 1));
		atomicAdd(&(hist_shared[bin]), 1);
	}
	__syncthreads();

	if (tid < NUM_BINS) {
		atomicAdd(&(hist[tid]), hist_shared[tid]);
	}
}



/// histogram
void histogram512(float *samples_h, size_t len_array, unsigned int **hist_h, float min_value, float max_value) {
	float *samples_d;
	unsigned int *hist_d;

	cudaError_t err = cudaMalloc((void**)&samples_d, len_array * sizeof(float));
	if (err != cudaSuccess) {
		std::cout << "Cuda Error allocating memory for samples_d: " << cudaGetErrorString(err) << std::endl;
		exit(-1);
	}

	err = cudaMalloc((void**)&hist_d, NUM_BINS * sizeof(unsigned int));
	if (err != cudaSuccess) {
		std::cout << "Cuda Error allocating memory for hist_d: " << cudaGetErrorString(err) << std::endl;
		cudaFree(samples_d);
		exit(-1);
	}

	err = cudaMemset(hist_d, 0, NUM_BINS * sizeof(unsigned int));
		if (err != cudaSuccess) {
			std::cout << "Cuda Error initialising hist_d: " << cudaGetErrorString(err) << std::endl;
			cudaFree(samples_d);
			cudaFree(hist_d);
			exit(-1);
	}

	err = cudaMemcpy(samples_d, samples_h, len_array * sizeof(float), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			std::cout << "Cuda Error copying samples to device: " << cudaGetErrorString(err) << std::endl;
			cudaFree(samples_d);
			cudaFree(hist_d);
			exit(-1);
	}

	dim3 block(BLOCK_SIZE);
	dim3 grid((len_array + BLOCK_SIZE - 1) / BLOCK_SIZE);
	histogramKernel512<<<grid, block>>>(samples_d, hist_d, len_array, min_value, max_value);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cout << "Cuda kernel launch error: " << cudaGetErrorString(err) << std::endl;
		cudaFree(samples_d);
		cudaFree(hist_d);
		exit(-1);
	}

	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		std::cout << "Cuda error during cudaDeviceSynchronize: " << cudaGetErrorString(err) << std::endl;
		cudaFree(samples_d);
		cudaFree(hist_d);
		exit(-1);
	}

	*hist_h = new unsigned int[NUM_BINS];

	err = cudaMemcpy(*hist_h, hist_d, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cout << "Cuda error copying histogram back to host: " << cudaGetErrorString(err) << std::endl;
		cudaFree(samples_d);
		cudaFree(hist_d);
		exit(-1);
	}

	cudaFree(samples_d);
	cudaFree(hist_d);
}
