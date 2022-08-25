#include "elastodynamics_cqbem_collocation_kernels.cuh"

#include "integrator_collocation.hpp"

#ifdef __CUDACC__
#include <cuda_runtime.h>
template <typename T> inline T exp(const T &val) { return thrust::exp(val); }
#else
#define __device__
#define __host__
template <typename T> inline T exp(const T &val) { return std::exp(val); }
#endif

using KernelScalarType = float;
using SingularityScalarType = double;

__device__ __host__ void compute_elastodynamic_B_trans_kernel_collocation(
    ComplexType *B_buffer, const ScalarType *V_buffer, const IntType *F_buffer, const ScalarType *N_buffer,
    Eigen::Index num_vertices, Eigen::Index num_faces, ScalarType _c1, ScalarType _c2, complex<double> _s,
    unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision, int i,
    const IntType *vertex_map_inverse
) {
    using ThrustComplexType = complex<ScalarType>;
    using ThrustMatrixX3c = Eigen::Matrix<ThrustComplexType, Eigen::Dynamic, 3, StorageOrder>;

    using KernelRowVector3s = Eigen::Matrix<KernelScalarType, 1, 3>;
    using SingularityRowVector3s = Eigen::Matrix<SingularityScalarType, 1, 3>;

    using KernelThrustComplexType = complex<KernelScalarType>;
    using KernelThrustMatrix3c = Eigen::Matrix<KernelThrustComplexType, 3, 3, StorageOrder>;

    using SingularityComplexType = complex<SingularityScalarType>;

    auto B = Eigen::Map<ThrustMatrixX3c>(reinterpret_cast<ThrustComplexType *>(B_buffer), num_vertices * 3, 3);
    auto V = Eigen::Map<const MatrixX3s>(V_buffer, num_vertices, (Eigen::Index)3);
    auto F = Eigen::Map<const MatrixX3i>(F_buffer, num_faces, (Eigen::Index)3);
    auto N = Eigen::Map<const MatrixX3s>(N_buffer, num_faces, (Eigen::Index)3);

    KernelThrustMatrix3c Bi = KernelThrustMatrix3c::Zero();

    const auto r_b_hat_ij = [&](const auto &r_hat, auto r, const auto &n, int i, int j) -> complex<decltype(r)> {
        using IntegrandScalarType = decltype(r);
        using IntegrandComplexType = complex<decltype(r)>;
        const IntegrandScalarType c1 = _c1, c2 = _c2;
        const IntegrandComplexType s = IntegrandComplexType(_s);

        constexpr auto kronecker_delta = [](int i, int j) -> IntegrandScalarType { return (i == j) ? 1.0 : 0.0; };
        const IntegrandComplexType exp_neg_rs_c1 = exp(-r * s / c1);
        const IntegrandComplexType exp_neg_rs_c2 = exp(-r * s / c2);

        return -(IntegrandScalarType)(1. / (4. * M_PI)) *
               (((r_hat[i] * n[j] + r_hat[j] * n[i]) / ((IntegrandScalarType)2. * r)) *
                    (((s * r / c2 + (IntegrandScalarType)1.) / (s * s)) * exp_neg_rs_c2 -
                     ((s * r / c1 + (IntegrandScalarType)1.) / (s * s)) * exp_neg_rs_c1) +
                kronecker_delta(i, j) * r_hat.dot(n) / r *
                    ((IntegrandScalarType)1. / (s * s) -
                     ((s * r / c2 + (IntegrandScalarType)1.) / (s * s)) * exp_neg_rs_c2));
    };

    const auto b_hat = [&](const auto &y, const auto &x, const auto &n) -> KernelThrustMatrix3c {
        KernelThrustMatrix3c mat;
        auto r_vec = y - x;
        const KernelRowVector3s r_hat = r_vec.stableNormalized();
        KernelScalarType r = r_vec.norm();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j <= i; j++) {
                mat(i, j) = r_b_hat_ij(r_hat, r, n, i, j) / r;
                if (i != j) mat(j, i) = mat(i, j);
            }
        }
        return mat;
    };

    const KernelRowVector3s x = V.row(i).cast<KernelScalarType>();
    for (Eigen::Index f_index = 0; f_index < num_faces; f_index++) {
        const Eigen::Index j1 = F(f_index, 0), j2 = F(f_index, 1), j3 = F(f_index, 2);
        bool has_singular_point = true;
        int singular_vert_index;
        if (i == j1 || (vertex_map_inverse != nullptr && vertex_map_inverse[i] == vertex_map_inverse[j1]))
            singular_vert_index = 0;
        else if (i == j2 || (vertex_map_inverse != nullptr && vertex_map_inverse[i] == vertex_map_inverse[j2]))
            singular_vert_index = 1;
        else if (i == j3 || (vertex_map_inverse != nullptr && vertex_map_inverse[i] == vertex_map_inverse[j3]))
            singular_vert_index = 2;
        else
            has_singular_point = false;

        if (!has_singular_point) {
            const KernelRowVector3s n = N.row(f_index).cast<KernelScalarType>();
            const KernelRowVector3s y1 = V.row(j1).cast<KernelScalarType>(), y2 = V.row(j2).cast<KernelScalarType>(),
                                    y3 = V.row(j3).cast<KernelScalarType>();
            Bi += integrate_collocation_nonsingular(
                [&](const auto &y) { return b_hat(y, x, n); }, y1, y2, y3, gaussian_quadrature_order,
                quadrature_subdivision
            );
        } else {
            const SingularityRowVector3s n = N.row(f_index).cast<SingularityScalarType>();
            const Eigen::Index j1 = F(f_index, singular_vert_index), j2 = F(f_index, (singular_vert_index + 1) % 3),
                               j3 = F(f_index, (singular_vert_index + 2) % 3);
            const SingularityRowVector3s y1 = V.row(j1).cast<SingularityScalarType>(),
                                         y2 = V.row(j2).cast<SingularityScalarType>(),
                                         y3 = V.row(j3).cast<SingularityScalarType>();

            integrate_collocation_weakly_singular<SingularityScalarType>(
                [&](auto func) { // update loop
                    for (int p = 0; p < 3; p++) {
                        for (int q = 0; q <= p; q++) {
                            SingularityComplexType elem = func(Eigen::Vector2i(p, q));
                            Bi(p, q) += elem;
                            if (p != q) Bi(q, p) += elem;
                        }
                    }
                },
                [&](const auto &y, const auto &r_hat, const auto &r,
                    const auto &indices) -> SingularityComplexType { // integrand
                    return r_b_hat_ij(r_hat, r, n, indices[0], indices[1]);
                },
                y1, y2, y3, gaussian_quadrature_order, quadrature_subdivision
            );
        }
    }

    B.middleRows<3>(3 * i) = Bi.cast<ThrustComplexType>();
}

#ifdef __CUDACC__
__global__ void compute_elastodynamic_B_trans_kernel_collocation_global_wrapper(
    ComplexType *B_buffer, const ScalarType *V_buffer, const IntType *F_buffer, const ScalarType *N_buffer,
    Eigen::Index num_vertices, Eigen::Index num_faces, ScalarType c1, ScalarType c2, complex<double> s,
    unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision, unsigned int thread_per_block,
    const IntType *vertex_map_inverse
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_vertices) {
        compute_elastodynamic_B_trans_kernel_collocation(
            B_buffer, V_buffer, F_buffer, N_buffer, num_vertices, num_faces, c1, c2, s, gaussian_quadrature_order,
            quadrature_subdivision, i, vertex_map_inverse
        );
    }
}
#endif
