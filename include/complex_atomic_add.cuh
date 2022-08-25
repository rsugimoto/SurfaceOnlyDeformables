#ifndef __COMPLEX_ATOMIC_ADD_CUH__
#define __COMPLEX_ATOMIC_ADD_CUH__

#ifdef __CUDACC__
#include <thrust/complex.h>
template <typename T> __inline__ __device__ void atomicAdd(thrust::complex<T> *address, thrust::complex<T> val) {
    atomicAdd((T *)address, val.real());
    atomicAdd(((T *)address) + 1, val.imag());
}
#endif

#endif //__COMPLEX_ATOMIC_ADD_CUH__