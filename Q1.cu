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
	//
	// Your code goes here.
	//
}


float findMaxValue(float* samples_h, size_t len_array){
	//
	// Your code goes here.
	//
	return 0.0f;
}


/////// Find Minimum
__global__ void minReduceKernel(float *d_in, size_t len_array){
	//
	// Your code goes here
	//
}


float findMinValue(float* samples_h, size_t len_array){
	//
	// Your code goes here
	//
	return 0.0f;
}



/////// Create Histogram
__global__ void histogramKernel512(float *d_in, unsigned int *hist, size_t len_array, float min_value, float max_value) {
	//
	// Your code goes here
	//
}



/// histogram
void histogram512(float *samples_h, size_t len_array, unsigned int **hist_h, float min_value, float max_value) {
	//
	// Your code goes here
	//
}
