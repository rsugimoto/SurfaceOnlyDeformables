#include "elastodynamics_cqbem_collocation_kernels.cuh"

#include "barycentric_coordinates.cuh"
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

__device__ __host__ void compute_elastodynamic_G_kernel_collocation(
    ComplexType *U_buffer, const ScalarType *V_buffer, const IntType *F_buffer, Eigen::Index num_vertices,
    Eigen::Index num_faces, ScalarType _c1, ScalarType _c2, ScalarType _rho, complex<double> _s,
    unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision, int i,
    const IntType *vertex_map_inverse
) {
    using ThrustComplexType = complex<ScalarType>;
    using ThrustMatrixXc = Eigen::Matrix<ThrustComplexType, Eigen::Dynamic, Eigen::Dynamic, StorageOrder>;

    using KernelRowVector3s = Eigen::Matrix<KernelScalarType, 1, 3>;
    using SingularityRowVector3s = Eigen::Matrix<SingularityScalarType, 1, 3>;

    using KernelThrustComplexType = complex<KernelScalarType>;
    using KernelThrustVector3c = Eigen::Matrix<KernelThrustComplexType, 3, 1>;
    using KernelThrustMatrix3c = Eigen::Matrix<KernelThrustComplexType, 3, 3, StorageOrder>;

    using SingularityComplexType = complex<SingularityScalarType>;
    using SingularityVector3c = Eigen::Matrix<SingularityComplexType, 3, 1>;

    auto U =
        Eigen::Map<ThrustMatrixXc>(reinterpret_cast<ThrustComplexType *>(U_buffer), num_vertices * 3, num_vertices * 3);
    auto V = Eigen::Map<const MatrixX3s>(V_buffer, num_vertices, (Eigen::Index)3);
    auto F = Eigen::Map<const MatrixX3i>(F_buffer, num_faces, (Eigen::Index)3);

    U.middleRows<3>(3 * i).setZero();

    const auto r_u_hat_ij = [&](const auto &r_hat, auto r, int i, int j) -> complex<decltype(r)> {
        using IntegrandScalarType = decltype(r);
        using IntegrandComplexType = complex<decltype(r)>;
        const IntegrandScalarType c1 = _c1, c2 = _c2, rho = _rho;
        const IntegrandComplexType s = IntegrandComplexType(_s);

        constexpr auto kronecker_delta = [](int i, int j) -> IntegrandScalarType { return (i == j) ? 1.0 : 0.0; };
        const IntegrandComplexType exp_neg_rs_c1 = exp(-r * s / c1);
        const IntegrandComplexType exp_neg_rs_c2 = exp(-r * s / c2);

        return ((IntegrandScalarType)(1. / (4. * M_PI)) / rho) *
               ((((IntegrandScalarType)3. * r_hat[i] * r_hat[j] - kronecker_delta(i, j)) / (r * r)) *
                    (((s * r / c1 + (IntegrandScalarType)1.) / (s * s)) * exp_neg_rs_c1 -
                     ((s * r / c2 + (IntegrandScalarType)1.) / (s * s)) * exp_neg_rs_c2) +
                (r_hat[i] * r_hat[j]) * (exp_neg_rs_c1 / (c1 * c1) - exp_neg_rs_c2 / (c2 * c2)) +
                (kronecker_delta(i, j) / (c2 * c2)) * exp_neg_rs_c2);
    };

    const auto u_hat = [&](const auto &y, const auto &x) -> KernelThrustMatrix3c {
        KernelThrustMatrix3c mat;
        auto r_vec = y - x;
        const KernelRowVector3s r_hat = r_vec.stableNormalized();
        KernelScalarType r = r_vec.norm();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j <= i; j++) {
                mat(i, j) = r_u_hat_ij(r_hat, r, i, j) / r;
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
            const KernelRowVector3s y1 = V.row(j1).cast<KernelScalarType>(), y2 = V.row(j2).cast<KernelScalarType>(),
                                    y3 = V.row(j3).cast<KernelScalarType>();
            const Eigen::Matrix<KernelThrustComplexType, 9, 3> mat = integrate_collocation_nonsingular(
                [&](const auto &y) {
                    KernelRowVector3s L;
                    barycentric_coordinates(y, y1, y2, y3, L);
                    L = L.cwiseMin(1.0).cwiseMax(0.0);
                    const auto _u_hat = u_hat(y, x);
                    Eigen::Matrix<KernelThrustComplexType, 9, 3> res;
                    for (int j = 0; j < 3; j++) res.middleRows<3>(3 * j) = _u_hat * L(j);
                    return res;
                },
                y1, y2, y3, gaussian_quadrature_order, quadrature_subdivision
            );
            U.block<3, 3>(3 * i, 3 * j1) += mat.block<3, 3>(0, 0).cast<ThrustComplexType>();
            U.block<3, 3>(3 * i, 3 * j2) += mat.block<3, 3>(3, 0).cast<ThrustComplexType>();
            U.block<3, 3>(3 * i, 3 * j3) += mat.block<3, 3>(6, 0).cast<ThrustComplexType>();
        } else {
            const Eigen::Index j1 = F(f_index, singular_vert_index), j2 = F(f_index, (singular_vert_index + 1) % 3),
                               j3 = F(f_index, (singular_vert_index + 2) % 3);
            const SingularityRowVector3s y1 = V.row(j1).cast<SingularityScalarType>(),
                                         y2 = V.row(j2).cast<SingularityScalarType>(),
                                         y3 = V.row(j3).cast<SingularityScalarType>();

            integrate_collocation_weakly_singular<SingularityScalarType>(
                [&](auto func) { // update loop
                    for (int p = 0; p < 3; p++) {
                        for (int q = 0; q <= p; q++) {
                            SingularityVector3c elems = func(Eigen::Vector2i(p, q));
                            U(3 * i + p, 3 * j1 + q) += elems(0);
                            U(3 * i + p, 3 * j2 + q) += elems(1);
                            U(3 * i + p, 3 * j3 + q) += elems(2);
                            if (p != q) {
                                U(3 * i + q, 3 * j1 + p) += elems(0);
                                U(3 * i + q, 3 * j2 + p) += elems(1);
                                U(3 * i + q, 3 * j3 + p) += elems(2);
                            }
                        }
                    }
                },
                [&](const auto &y, const auto &r_hat, const auto &r,
                    const auto &indices) -> SingularityVector3c { // integrand
                    SingularityRowVector3s L;
                    barycentric_coordinates(y, y1, y2, y3, L);
                    L = L.cwiseMin(1.0).cwiseMax(0.0);
                    return r_u_hat_ij(r_hat, r, indices[0], indices[1]) * L.cast<SingularityComplexType>();
                },
                y1, y2, y3, gaussian_quadrature_order, quadrature_subdivision
            );
        }
    }
}

#ifdef __CUDACC__
__global__ void compute_elastodynamic_G_kernel_collocation_global_wrapper(
    ComplexType *U_buffer, const ScalarType *V_buffer, const IntType *F_buffer, Eigen::Index num_vertices,
    Eigen::Index num_faces, ScalarType c1, ScalarType c2, ScalarType rho, complex<double> s,
    unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision, unsigned int thread_per_block,
    const IntType *vertex_map_inverse
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_vertices) {
        compute_elastodynamic_G_kernel_collocation(
            U_buffer, V_buffer, F_buffer, num_vertices, num_faces, c1, c2, rho, s, gaussian_quadrature_order,
            quadrature_subdivision, i, vertex_map_inverse
        );
    }
}
#endif