#ifndef __ELASTODYNAMICS_CQBEM_COLLOCATION_KERNELS_CUH__
#define __ELASTODYNAMICS_CQBEM_COLLOCATION_KERNELS_CUH__

#include "type_declaration.hpp"

#ifdef __CUDACC__
#include <thrust/complex.h>
template <class T> using complex = thrust::complex<T>;
#else
#define __device__
#define __host__
template <class T> using complex = std::complex<T>;
#endif

__device__ __host__ void compute_elastodynamic_H_kernel_collocation(
    ComplexType *P_buffer, const ScalarType *V_buffer, const IntType *F_buffer, const ScalarType *N_buffer,
    Eigen::Index num_vertices, Eigen::Index num_faces, ScalarType _c1, ScalarType _c2, complex<double> _s,
    unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision, int i,
    const IntType *vertex_map_inverse
);

#ifdef __CUDACC__
__global__ void compute_elastodynamic_H_kernel_collocation_global_wrapper(
    ComplexType *P_buffer, const ScalarType *V_buffer, const IntType *F_buffer, const ScalarType *N_buffer,
    Eigen::Index num_vertices, Eigen::Index num_faces, ScalarType c1, ScalarType c2, complex<double> s,
    unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision, unsigned int thread_per_block,
    const IntType *vertex_map_inverse
);
#endif

__device__ __host__ void compute_elastodynamic_G_kernel_collocation(
    ComplexType *U_buffer, const ScalarType *V_buffer, const IntType *F_buffer, Eigen::Index num_vertices,
    Eigen::Index num_faces, ScalarType _c1, ScalarType _c2, ScalarType _rho, complex<double> _s,
    unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision, int i,
    const IntType *vertex_map_inverse
);

#ifdef __CUDACC__
__global__ void compute_elastodynamic_G_kernel_collocation_global_wrapper(
    ComplexType *U_buffer, const ScalarType *V_buffer, const IntType *F_buffer, Eigen::Index num_vertices,
    Eigen::Index num_faces, ScalarType c1, ScalarType c2, ScalarType rho, complex<double> s,
    unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision, unsigned int thread_per_block,
    const IntType *vertex_map_inverse
);
#endif

__device__ __host__ void compute_elastodynamic_B_trans_kernel_collocation(
    ComplexType *B_buffer, const ScalarType *V_buffer, const IntType *F_buffer, const ScalarType *N_buffer,
    Eigen::Index num_vertices, Eigen::Index num_faces, ScalarType _c1, ScalarType _c2, complex<double> _s,
    unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision, int i,
    const IntType *vertex_map_inverse
);

#ifdef __CUDACC__
__global__ void compute_elastodynamic_B_trans_kernel_collocation_global_wrapper(
    ComplexType *B_buffer, const ScalarType *V_buffer, const IntType *F_buffer, const ScalarType *N_buffer,
    Eigen::Index num_vertices, Eigen::Index num_faces, ScalarType c1, ScalarType c2, complex<double> s,
    unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision, unsigned int thread_per_block,
    const IntType *vertex_map_inverse
);
#endif

__device__ __host__ void compute_elastodynamic_B_angular_kernel_collocation(
    ComplexType *B_buffer, const ScalarType *V_buffer, const IntType *F_buffer, const ScalarType *N_buffer,
    Eigen::Index num_vertices, Eigen::Index num_faces, ScalarType _c1, ScalarType _c2, complex<double> _s,
    unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision, int i,
    const IntType *vertex_map_inverse
);

#ifdef __CUDACC__
__global__ void compute_elastodynamic_B_angular_kernel_collocation_global_wrapper(
    ComplexType *B_buffer, const ScalarType *V_buffer, const IntType *F_buffer, const ScalarType *N_buffer,
    Eigen::Index num_vertices, Eigen::Index num_faces, ScalarType c1, ScalarType c2, complex<double> s,
    unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision, unsigned int thread_per_block,
    const IntType *vertex_map_inverse
);
#endif

#endif //!__ELASTODYNAMICS_CQBEM_COLLOCATION_KERNELS_CUH__
