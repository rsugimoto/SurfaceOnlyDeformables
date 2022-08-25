#ifndef __CUDA_CHECK_ERROR_HPP__
#define __CUDA_CHECK_ERROR_HPP__

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <iostream>

template <class Func> inline void cuda_check_error(Func func) {
    cudaError_t error = func();
    if (error != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
    }
}

inline void cuda_check_last_error() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
    }
}

#endif

#endif //__CUDA_CHECK_ERROR_HPP__