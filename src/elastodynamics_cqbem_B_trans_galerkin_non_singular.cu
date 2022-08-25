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

__device__ __host__ void compute_elastodynamic_B_trans_kernel_galerkin_non_singular(
    ComplexType *B_buffer, const ScalarType *V_buffer, const IntType *F_buffer, const ScalarType *N_buffer,
    Eigen::Index num_vertices, Eigen::Index num_faces, NonSingularScalarType c1, NonSingularScalarType c2,
    complex<NonSingularScalarType> s, unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision,
    int f1, int f2, const IntType *vertex_map_inverse
) {
    auto B =
        Eigen::Map<ThrustMatrixXc>(reinterpret_cast<ThrustComplexType *>(B_buffer), num_vertices * 3, (Eigen::Index)3);
    const auto V = Eigen::Map<const MatrixX3s>(V_buffer, num_vertices, (Eigen::Index)3);
    const auto F = Eigen::Map<const MatrixX3i>(F_buffer, num_faces, (Eigen::Index)3);
    const auto N = Eigen::Map<const MatrixX3s>(N_buffer, num_faces, (Eigen::Index)3);

    const auto b_hat_ij = [&](const auto &r_hat, auto r, const auto &n, int i, int j) -> complex<decltype(r)> {
        constexpr auto kronecker_delta = [](int i, int j) -> NonSingularScalarType { return (i == j) ? 1.0 : 0.0; };
        const NonSingularComplexType exp_neg_rs_c1 = exp(-r * s / c1);
        const NonSingularComplexType exp_neg_rs_c2 = exp(-r * s / c2);

        return -(NonSingularScalarType)(1. / (4. * M_PI)) *
               (((r_hat[i] * n[j] + r_hat[j] * n[i]) / ((NonSingularScalarType)2. * r)) *
                    (((s * r / c2 + (NonSingularScalarType)1.) / (s * s)) * exp_neg_rs_c2 -
                     ((s * r / c1 + (NonSingularScalarType)1.) / (s * s)) * exp_neg_rs_c1) +
                kronecker_delta(i, j) * r_hat.dot(n) / r *
                    ((NonSingularScalarType)1. / (s * s) -
                     ((s * r / c2 + (NonSingularScalarType)1.) / (s * s)) * exp_neg_rs_c2)) /
               r;
    };

    const auto b_hat = [&](const auto &y, const auto &x, const auto &n) {
        const auto r_vec = y - x;
        const auto r_hat = r_vec.stableNormalized().eval();
        const auto r = r_vec.norm();
        Eigen::Matrix<NonSingularComplexType, 3, 3> mat;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j <= i; j++) {
                mat(i, j) = b_hat_ij(r_hat, r, n, i, j);
                if (i != j) mat(j, i) = mat(i, j);
            }
        }
        return mat;
    };

    const Eigen::Matrix<NonSingularScalarType, 1, 3> n = N.row(f2).cast<NonSingularScalarType>();
    const auto integrand = [&](const auto &x, const auto &y, const auto &Lx, const auto &Ly) {
        auto _b_hat = b_hat(y, x, n);
        Eigen::Matrix<NonSingularComplexType, 9, 3, Eigen::RowMajor> res;
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
    for (int i = 0; i < 3; i++) {
#ifdef __CUDA_ARCH__
        for (int p = 0; p < 3; p++) {
            for (int q = 0; q < 3; q++) {
                atomicAdd((ThrustComplexType *)&(B(3 * j1[i] + p, q)), (ThrustComplexType)mat(3 * i + p, q));
            }
        }
#else
        B.middleRows<3>(3 * j1[i]) += mat.middleRows<3>(3 * i).cast<ThrustComplexType>();
#endif
    }
}

#ifdef __CUDACC__
__global__ void compute_elastodynamic_B_trans_kernel_galerkin_non_singular_global_wrapper(
    ComplexType *B_buffer, const ScalarType *V_buffer, const IntType *F_buffer, const ScalarType *N_buffer,
    Eigen::Index num_vertices, Eigen::Index num_faces, NonSingularScalarType c1, NonSingularScalarType c2,
    complex<NonSingularScalarType> s, unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision,
    const IntType *vertex_map_inverse
) {
    int f1 = blockIdx.x * blockDim.x + threadIdx.x;
    int f2 = blockIdx.y * blockDim.y + threadIdx.y;
    if (f1 < num_faces && f2 < num_faces)
        compute_elastodynamic_B_trans_kernel_galerkin_non_singular(
            B_buffer, V_buffer, F_buffer, N_buffer, num_vertices, num_faces, c1, c2, s, gaussian_quadrature_order,
            quadrature_subdivision, f1, f2, vertex_map_inverse
        );
}
#endif
