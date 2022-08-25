#ifndef __ELASTODYNAMICS_CQBEM_GALERKIN_KERNELS_CUH__
#define __ELASTODYNAMICS_CQBEM_GALERKIN_KERNELS_CUH__

#include "type_declaration.hpp"

#ifdef __CUDACC__
#include <thrust/complex.h>
template <class T> using complex = thrust::complex<T>;
#else
#define __device__
#define __host__
template <class T> using complex = std::complex<T>;
#endif

using NonSingularScalarType = float;
using SingularScalarType = double;

__device__ __host__ void compute_elastodynamic_G_kernel_galerkin_non_singular(
    ComplexType *U_buffer, const ScalarType *V_buffer, const IntType *F_buffer, Eigen::Index num_vertices,
    Eigen::Index num_faces, NonSingularScalarType c1, NonSingularScalarType c2, NonSingularScalarType rho,
    complex<NonSingularScalarType> s, unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision,
    int f1, int f2, const IntType *vertex_map_inverse
);

#ifdef __CUDACC__
__global__ void compute_elastodynamic_G_kernel_galerkin_non_singular_global_wrapper(
    ComplexType *U_buffer, const ScalarType *V_buffer, const IntType *F_buffer, Eigen::Index num_vertices,
    Eigen::Index num_faces, NonSingularScalarType c1, NonSingularScalarType c2, NonSingularScalarType rho,
    complex<NonSingularScalarType> s, unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision,
    const IntType *vertex_map_inverse
);
#endif

void compute_elastodynamic_G_kernel_galerkin_singular(
    ComplexType *U_buffer, const ScalarType *V_buffer, const IntType *F_buffer, Eigen::Index num_vertices,
    Eigen::Index num_faces, SingularScalarType c1, SingularScalarType c2, SingularScalarType rho,
    std::complex<SingularScalarType> s, unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision,
    const IntType *vertex_map_inverse
);

__device__ __host__ void compute_elastodynamic_H_kernel_galerkin_non_singular(
    ComplexType *P_buffer, const ScalarType *V_buffer, const IntType *F_buffer, const ScalarType *N_buffer,
    Eigen::Index num_vertices, Eigen::Index num_faces, NonSingularScalarType c1, NonSingularScalarType c2,
    complex<NonSingularScalarType> s, unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision,
    int f1, int f2, const IntType *vertex_map_inverse
);

#ifdef __CUDACC__
__global__ void compute_elastodynamic_H_kernel_galerkin_non_singular_global_wrapper(
    ComplexType *P_buffer, const ScalarType *V_buffer, const IntType *F_buffer, const ScalarType *N_buffer,
    Eigen::Index num_vertices, Eigen::Index num_faces, NonSingularScalarType c1, NonSingularScalarType c2,
    complex<NonSingularScalarType> s, unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision,
    const IntType *vertex_map_inverse
);
#endif

void compute_elastodynamic_H_kernel_galerkin_singular(
    ComplexType *U_buffer, const ScalarType *P_buffer, const IntType *F_buffer, const ScalarType *N_buffer,
    Eigen::Index num_vertices, Eigen::Index num_faces, SingularScalarType c1, SingularScalarType c2,
    std::complex<SingularScalarType> s, unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision,
    const IntType *vertex_map_inverse
);

__device__ __host__ void compute_elastodynamic_B_trans_kernel_galerkin_non_singular(
    ComplexType *B_buffer, const ScalarType *V_buffer, const IntType *F_buffer, const ScalarType *N_buffer,
    Eigen::Index num_vertices, Eigen::Index num_faces, NonSingularScalarType c1, NonSingularScalarType c2,
    complex<NonSingularScalarType> s, unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision,
    int f1, int f2, const IntType *vertex_map_inverse
);

#ifdef __CUDACC__
__global__ void compute_elastodynamic_B_trans_kernel_galerkin_non_singular_global_wrapper(
    ComplexType *B_buffer, const ScalarType *V_buffer, const IntType *F_buffer, const ScalarType *N_buffer,
    Eigen::Index num_vertices, Eigen::Index num_faces, NonSingularScalarType c1, NonSingularScalarType c2,
    complex<NonSingularScalarType> s, unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision,
    const IntType *vertex_map_inverse
);
#endif

void compute_elastodynamic_B_trans_kernel_galerkin_singular(
    ComplexType *B_buffer, const ScalarType *V_buffer, const IntType *F_buffer, const ScalarType *N_buffer,
    Eigen::Index num_vertices, Eigen::Index num_faces, SingularScalarType c1, SingularScalarType c2,
    std::complex<SingularScalarType> s, unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision,
    const IntType *vertex_map_inverse
);

__device__ __host__ void compute_elastodynamic_B_angular_kernel_galerkin_non_singular(
    ComplexType *B_buffer, const ScalarType *V_buffer, const IntType *F_buffer, const ScalarType *N_buffer,
    Eigen::Index num_vertices, Eigen::Index num_faces, NonSingularScalarType c1, NonSingularScalarType c2,
    complex<NonSingularScalarType> s, unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision,
    int f1, int f2, const IntType *vertex_map_inverse
);

#ifdef __CUDACC__
__global__ void compute_elastodynamic_B_angular_kernel_galerkin_non_singular_global_wrapper(
    ComplexType *B_buffer, const ScalarType *V_buffer, const IntType *F_buffer, const ScalarType *N_buffer,
    Eigen::Index num_vertices, Eigen::Index num_faces, NonSingularScalarType c1, NonSingularScalarType c2,
    complex<NonSingularScalarType> s, unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision,
    const IntType *vertex_map_inverse
);
#endif

void compute_elastodynamic_B_angular_kernel_galerkin_singular(
    ComplexType *B_buffer, const ScalarType *V_buffer, const IntType *F_buffer, const ScalarType *N_buffer,
    Eigen::Index num_vertices, Eigen::Index num_faces, SingularScalarType c1, SingularScalarType c2,
    std::complex<SingularScalarType> s, unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision,
    const IntType *vertex_map_inverse
);

#endif //!__ELASTODYNAMICS_CQBEM_GALERKIN_KERNELS_CUH__
