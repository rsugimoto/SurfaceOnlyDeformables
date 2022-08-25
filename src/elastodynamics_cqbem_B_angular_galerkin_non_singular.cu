#include "elastodynamics_cqbem_galerkin_kernels.cuh"

#include "complex_atomic_add.cuh"
#include "integrator_galerkin.hpp"

#ifdef __CUDACC__
#include <cuda_runtime.h>
template <typename T> inline T exp(const T &val) { return thrust::exp(val); }
#else
#define __device__
#define __host__
template <typename T> inline T exp(const T &val) { return std::exp(val); }
#endif

using ThrustComplexType = complex<ScalarType>;
using ThrustMatrixXc = Eigen::Matrix<ThrustComplexType, Eigen::Dynamic, Eigen::Dynamic, StorageOrder>;

using NonSingularComplexType = complex<NonSingularScalarType>;

__device__ __host__ void compute_elastodynamic_B_angular_kernel_galerkin_non_singular(
    ComplexType *B_buffer, const ScalarType *V_buffer, const IntType *F_buffer, const ScalarType *N_buffer,
    Eigen::Index num_vertices, Eigen::Index num_faces, NonSingularScalarType c1, NonSingularScalarType c2,
    complex<NonSingularScalarType> s, unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision,
    int f1, int f2, const IntType *vertex_map_inverse
) {
    const auto V = Eigen::Map<const MatrixX3s>(V_buffer, num_vertices, (Eigen::Index)3);
    const auto F = Eigen::Map<const MatrixX3i>(F_buffer, num_faces, (Eigen::Index)3);
    const auto N = Eigen::Map<const MatrixX3s>(N_buffer, num_faces, (Eigen::Index)3);

    auto ind_3d = [](auto *tensor, int _dim1, int _dim2, int _dim3, int _ind1, int _ind2, int _ind3) -> auto & {
        return tensor[_dim3 * _dim2 * _ind1 + _dim3 * _ind2 + _ind3]; // Row major
        // return tensor[_dim1*_dim2*_ind3 + _dim1*_ind2 + _ind1]; //Column major
    };

    const auto b_hat_ijk = [&](const auto &r_hat, auto r, const auto &n, int i, int j, int k) -> complex<decltype(r)> {
        constexpr auto kronecker_delta = [](int i, int j) -> NonSingularScalarType { return (i == j) ? 1.0 : 0.0; };
        const NonSingularComplexType exp_neg_rs_c1 = exp(-r * s / c1);
        const NonSingularComplexType exp_neg_rs_c2 = exp(-r * s / c2);

        return -(NonSingularScalarType)(1. / (4. * M_PI)) *
               ((r_hat[i] * r_hat[k] * n[j] + r_hat[j] * r_hat[k] * n[i] + kronecker_delta(j, k) * n[i] +
                 kronecker_delta(i, k) * n[j]) /
                    (NonSingularScalarType)2. *
                    (((s * r / c2 + (NonSingularScalarType)1.) / (s * s)) * exp_neg_rs_c2 -
                     ((s * r / c1 + (NonSingularScalarType)1.) / (s * s)) * exp_neg_rs_c1) /
                    r -
                (kronecker_delta(j, k) * n(i) + kronecker_delta(i, k) * n(j)) / ((NonSingularScalarType)2. * s) *
                    (exp_neg_rs_c2 / c2 - exp_neg_rs_c1 / c1) -
                (kronecker_delta(i, j) * n(k)) / (c2 * s) * exp_neg_rs_c2);
    };

    const auto b_hat = [&](const auto &y, const auto &x, const auto &n) {
        const auto r_vec = y - x;
        const auto r_hat = r_vec.stableNormalized().eval();
        const auto r = r_vec.norm();
        Eigen::Matrix<NonSingularComplexType, 3, 9> mat;
        for (int k = 0; k < 3; k++) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j <= i; j++) {
                    mat(i, 3 * k + j) = b_hat_ijk(r_hat, r, n, i, j, k);
                    if (i != j) mat(j, 3 * k + i) = mat(i, 3 * k + j);
                }
            }
        }
        return mat;
    };

    const Eigen::Matrix<NonSingularScalarType, 1, 3> n = N.row(f2).cast<NonSingularScalarType>();
    const auto integrand = [&](const auto &x, const auto &y, const auto &Lx, const auto &Ly) {
        auto _b_hat = b_hat(y, x, n);
        Eigen::Matrix<NonSingularComplexType, 9, 9, Eigen::RowMajor> res;
        for (int i = 0; i < 3; i++) res.middleRows<3>(3 * i) = Lx(i) * _b_hat;
        return res;
    };

    // if singular, immediately return
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (F(f1, i) == F(f2, j) ||
                (vertex_map_inverse != nullptr && vertex_map_inverse[F(f1, i)] == vertex_map_inverse[F(f2, j)]))
                return;

    // non-singular
    const RowVector3i j1 = F.row(f1), j2 = F.row(f2);
    const auto mat = integrate_galerkin_nonsingular(
        integrand, V.row(j1[0]).cast<NonSingularScalarType>().eval(), V.row(j1[1]).cast<NonSingularScalarType>().eval(),
        V.row(j1[2]).cast<NonSingularScalarType>().eval(), V.row(j2[0]).cast<NonSingularScalarType>().eval(),
        V.row(j2[1]).cast<NonSingularScalarType>().eval(), V.row(j2[2]).cast<NonSingularScalarType>().eval(),
        gaussian_quadrature_order, quadrature_subdivision
    );

#pragma omp critical
    for (int k = 0; k < 3; k++) {
        for (int i = 0; i < 3; i++) {
            for (int p = 0; p < 3; p++) {
                for (int q = 0; q < 3; q++) {
#ifdef __CUDA_ARCH__
                    atomicAdd(
                        &(ind_3d((ThrustComplexType *)B_buffer, num_vertices * 3, 3, 3, 3 * j1[i] + p, q, k)),
                        (ThrustComplexType)mat(3 * i + p, 3 * k + q)
                    );
#else
                    ind_3d((ThrustComplexType *)B_buffer, num_vertices * 3, 3, 3, 3 * j1[i] + p, q, k) +=
                        (ThrustComplexType)mat(3 * i + p, 3 * k + q);
#endif
                }
            }
        }
    }
}

#ifdef __CUDACC__
__global__ void compute_elastodynamic_B_angular_kernel_galerkin_non_singular_global_wrapper(
    ComplexType *B_buffer, const ScalarType *V_buffer, const IntType *F_buffer, const ScalarType *N_buffer,
    Eigen::Index num_vertices, Eigen::Index num_faces, NonSingularScalarType c1, NonSingularScalarType c2,
    complex<NonSingularScalarType> s, unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision,
    const IntType *vertex_map_inverse
) {
    int f1 = blockIdx.x * blockDim.x + threadIdx.x;
    int f2 = blockIdx.y * blockDim.y + threadIdx.y;
    if (f1 < num_faces && f2 < num_faces)
        compute_elastodynamic_B_angular_kernel_galerkin_non_singular(
            B_buffer, V_buffer, F_buffer, N_buffer, num_vertices, num_faces, c1, c2, s, gaussian_quadrature_order,
            quadrature_subdivision, f1, f2, vertex_map_inverse
        );
}
#endif
