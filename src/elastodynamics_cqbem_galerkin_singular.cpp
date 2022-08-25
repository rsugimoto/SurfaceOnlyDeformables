#include "elastodynamics_cqbem_galerkin_kernels.cuh"

#include "integrator_galerkin.hpp"

using SingularComplexType = std::complex<SingularScalarType>;

// All singular integrals are evaluated on CPU. As the computation is done in single precision, there is no gain in
// speed with GPU, and CUDA compilation takes too long.

void compute_elastodynamic_G_kernel_galerkin_singular(
    ComplexType *U_buffer, const ScalarType *V_buffer, const IntType *F_buffer, Eigen::Index num_vertices,
    Eigen::Index num_faces, SingularScalarType c1, SingularScalarType c2, SingularScalarType rho,
    std::complex<SingularScalarType> s, unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision,
    const IntType *vertex_map_inverse
) {
    auto U = Eigen::Map<MatrixXc>(U_buffer, num_vertices * 3, num_vertices * 3);
    const auto V = Eigen::Map<const MatrixX3s>(V_buffer, num_vertices, (Eigen::Index)3);
    const auto F = Eigen::Map<const MatrixX3i>(F_buffer, num_faces, (Eigen::Index)3);

    const auto u_hat_ij = [&](const auto &r_hat, auto r, int i, int j) -> std::complex<decltype(r)> {
        constexpr auto kronecker_delta = [](int i, int j) -> SingularScalarType { return (i == j) ? 1.0 : 0.0; };
        const SingularComplexType exp_neg_rs_c1 = std::exp(-r * s / c1);
        const SingularComplexType exp_neg_rs_c2 = std::exp(-r * s / c2);

        return ((SingularScalarType)(1. / (4. * M_PI)) / rho) *
               ((((SingularScalarType)3. * r_hat[i] * r_hat[j] - kronecker_delta(i, j)) / (r * r)) *
                    (((s * r / c1 + (SingularScalarType)1.) / (s * s)) * exp_neg_rs_c1 -
                     ((s * r / c2 + (SingularScalarType)1.) / (s * s)) * exp_neg_rs_c2) +
                (r_hat[i] * r_hat[j]) * (exp_neg_rs_c1 / (c1 * c1) - exp_neg_rs_c2 / (c2 * c2)) +
                (kronecker_delta(i, j) / (c2 * c2)) * exp_neg_rs_c2) /
               r;
    };

    const auto u_hat = [&](const auto &y, const auto &x) {
        const auto r_vec = y - x;
        const auto r_hat = r_vec.stableNormalized().eval();
        const auto r = r_vec.norm();
        Eigen::Matrix<SingularComplexType, 3, 3> mat;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j <= i; j++) {
                mat(i, j) = u_hat_ij(r_hat, r, i, j);
                if (i != j) mat(j, i) = mat(i, j);
            }
        }
        return mat;
    };

    const auto integrand = [&](const auto &x, const auto &y, const auto &Lx, const auto &Ly) {
        auto _u_hat = u_hat(y, x);
        Eigen::Matrix<SingularComplexType, 9, 9, Eigen::RowMajor> res;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) { res.block<3, 3>(3 * i, 3 * j) = Lx(i) * Ly(j) * _u_hat; }
        }
        return res;
    };

#pragma omp parallel for
    for (int f1 = 0; f1 < num_faces; f1++) {
        for (int f2 = 0; f2 < num_faces; f2++) {
            Eigen::Matrix<SingularComplexType, 9, 9, Eigen::RowMajor> mat;
            Eigen::Index j1[3], j2[3];
            bool is_singular = integrate_galerkin_singular<SingularScalarType>(
                integrand, mat, F, V, f1, f2, j1, j2, gaussian_quadrature_order, quadrature_subdivision,
                vertex_map_inverse
            );

            if (is_singular) {
#pragma omp critical
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        U.block<3, 3>(3 * j1[i], 3 * j2[j]) += mat.block<3, 3>(3 * i, 3 * j).cast<ComplexType>();
                    }
                }
            }
        }
    }
}

void compute_elastodynamic_H_kernel_galerkin_singular(
    ComplexType *P_buffer, const ScalarType *V_buffer, const IntType *F_buffer, const ScalarType *N_buffer,
    Eigen::Index num_vertices, Eigen::Index num_faces, SingularScalarType c1, SingularScalarType c2,
    std::complex<SingularScalarType> s, unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision,
    const IntType *vertex_map_inverse
) {
    auto P = Eigen::Map<MatrixXc>(P_buffer, num_vertices * 3, num_vertices * 3);
    const auto V = Eigen::Map<const MatrixX3s>(V_buffer, num_vertices, (Eigen::Index)3);
    const auto F = Eigen::Map<const MatrixX3i>(F_buffer, num_faces, (Eigen::Index)3);
    const auto N = Eigen::Map<const MatrixX3s>(N_buffer, num_faces, (Eigen::Index)3);

    const auto p_hat_ij = [&](const auto &r_hat, auto r, const auto &n, int i, int j) -> std::complex<decltype(r)> {
        constexpr auto kronecker_delta = [](int i, int j) -> SingularScalarType { return (i == j) ? 1.0 : 0.0; };
        const SingularComplexType exp_neg_rs_c1 = std::exp(-r * s / c1);
        const SingularComplexType exp_neg_rs_c2 = std::exp(-r * s / c2);

        const SingularScalarType r_hat_dot_n = r_hat.dot(n);
        return (SingularScalarType)(1. / (4. * M_PI)) *
               (((SingularScalarType)6. * c2 * c2 / (r * r)) *
                    (r_hat[i] * n[j] + r_hat[j] * n[i] +
                     (kronecker_delta(i, j) - (SingularScalarType)5. * r_hat[i] * r_hat[j]) * r_hat_dot_n) *
                    (((exp_neg_rs_c1 / (r * s)) * ((SingularScalarType)1. / c1 + (SingularScalarType)1. / (r * s)) -
                      (exp_neg_rs_c2 / (r * s)) * ((SingularScalarType)1. / c2 + (SingularScalarType)1. / (r * s)))) +
                (exp_neg_rs_c1 / (r * r)) *
                    ((SingularScalarType)2. * c2 * c2 / (c1 * c1) *
                         ((SingularScalarType)2. * r_hat[i] * n[j] + r_hat[j] * n[i] -
                          ((SingularScalarType)6. * r_hat[i] * r_hat[j] - kronecker_delta(i, j)) * r_hat_dot_n) -
                     r_hat[i] * n[j]) +
                (exp_neg_rs_c2 / (r * r)) *
                    ((SingularScalarType)12. * r_hat[i] * r_hat[j] * r_hat_dot_n -
                     (SingularScalarType)2. * r_hat[i] * n[j] - (SingularScalarType)3. * r_hat[j] * n[i] -
                     (SingularScalarType)3. * kronecker_delta(i, j) * r_hat_dot_n) -
                (exp_neg_rs_c1 * s / (r * c1)) *
                    (r_hat[i] * n[j] + (SingularScalarType)2. * c2 * c2 / (c1 * c1) *
                                           (r_hat[i] * r_hat[j] * r_hat_dot_n - r_hat[i] * n[j])) +
                (exp_neg_rs_c2 * s / (r * c2)) * ((SingularScalarType)2. * r_hat[i] * r_hat[j] * r_hat_dot_n -
                                                  kronecker_delta(i, j) * r_hat_dot_n - r_hat[j] * n[i]));
    };

    const auto p_hat = [&](const auto &y, const auto &x, const auto &n) {
        const auto r_vec = y - x;
        const auto r_hat = r_vec.stableNormalized().eval();
        const auto r = r_vec.norm();
        Eigen::Matrix<SingularComplexType, 3, 3> mat;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) { mat(i, j) = p_hat_ij(r_hat, r, n, i, j); }
        }
        return mat;
    };

#pragma omp parallel for
    for (int f1 = 0; f1 < num_faces; f1++) {
        for (int f2 = 0; f2 < num_faces; f2++) {
            const Eigen::Matrix<SingularScalarType, 1, 3> n = N.row(f2).cast<SingularScalarType>();
            const auto integrand = [&](const auto &x, const auto &y, const auto &Lx, const auto &Ly) {
                auto _p_hat = p_hat(y, x, n);
                Eigen::Matrix<SingularComplexType, 9, 9, Eigen::RowMajor> res;
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) { res.block<3, 3>(3 * i, 3 * j) = Lx(i) * Ly(j) * _p_hat; }
                }
                return res;
            };

            Eigen::Matrix<SingularComplexType, 9, 9, Eigen::RowMajor> mat;
            Eigen::Index j1[3], j2[3];
            bool is_singular = integrate_galerkin_singular<SingularScalarType>(
                integrand, mat, F, V, f1, f2, j1, j2, gaussian_quadrature_order, quadrature_subdivision,
                vertex_map_inverse
            );

            if (is_singular) {
#pragma omp critical
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        P.block<3, 3>(3 * j1[i], 3 * j2[j]) += mat.block<3, 3>(3 * i, 3 * j).cast<ComplexType>();
                    }
                }
            }
        }
    }
}

void compute_elastodynamic_B_trans_kernel_galerkin_singular(
    ComplexType *B_buffer, const ScalarType *V_buffer, const IntType *F_buffer, const ScalarType *N_buffer,
    Eigen::Index num_vertices, Eigen::Index num_faces, SingularScalarType c1, SingularScalarType c2,
    std::complex<SingularScalarType> s, unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision,
    const IntType *vertex_map_inverse
) {
    auto B = Eigen::Map<MatrixXc>(B_buffer, num_vertices * 3, (Eigen::Index)3);
    const auto V = Eigen::Map<const MatrixX3s>(V_buffer, num_vertices, (Eigen::Index)3);
    const auto F = Eigen::Map<const MatrixX3i>(F_buffer, num_faces, (Eigen::Index)3);
    const auto N = Eigen::Map<const MatrixX3s>(N_buffer, num_faces, (Eigen::Index)3);

    const auto b_hat_ij = [&](const auto &r_hat, auto r, const auto &n, int i, int j) -> std::complex<decltype(r)> {
        constexpr auto kronecker_delta = [](int i, int j) -> SingularScalarType { return (i == j) ? 1.0 : 0.0; };
        const SingularComplexType exp_neg_rs_c1 = std::exp(-r * s / c1);
        const SingularComplexType exp_neg_rs_c2 = std::exp(-r * s / c2);

        return -(SingularScalarType)(1. / (4. * M_PI)) *
               (((r_hat[i] * n[j] + r_hat[j] * n[i]) / ((SingularScalarType)2. * r)) *
                    (((s * r / c2 + (SingularScalarType)1.) / (s * s)) * exp_neg_rs_c2 -
                     ((s * r / c1 + (SingularScalarType)1.) / (s * s)) * exp_neg_rs_c1) +
                kronecker_delta(i, j) * r_hat.dot(n) / r *
                    ((SingularScalarType)1. / (s * s) -
                     ((s * r / c2 + (SingularScalarType)1.) / (s * s)) * exp_neg_rs_c2)) /
               r;
    };

    const auto b_hat = [&](const auto &y, const auto &x, const auto &n) {
        const auto r_vec = y - x;
        const auto r_hat = r_vec.stableNormalized().eval();
        const auto r = r_vec.norm();
        Eigen::Matrix<SingularComplexType, 3, 3> mat;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j <= i; j++) {
                mat(i, j) = b_hat_ij(r_hat, r, n, i, j);
                if (i != j) mat(j, i) = mat(i, j);
            }
        }
        return mat;
    };

#pragma omp parallel for
    for (int f1 = 0; f1 < num_faces; f1++) {
        for (int f2 = 0; f2 < num_faces; f2++) {
            const Eigen::Matrix<SingularScalarType, 1, 3> n = N.row(f2).cast<SingularScalarType>();
            const auto integrand = [&](const auto &x, const auto &y, const auto &Lx, const auto &Ly) {
                auto _b_hat = b_hat(y, x, n);
                Eigen::Matrix<SingularComplexType, 9, 3, Eigen::RowMajor> res;
                for (int i = 0; i < 3; i++) res.middleRows<3>(3 * i) = Lx(i) * _b_hat;
                return res;
            };

            Eigen::Matrix<SingularComplexType, 9, 3, Eigen::RowMajor> mat;
            Eigen::Index j1[3], j2[3];
            bool is_singular = integrate_galerkin_singular<SingularScalarType>(
                integrand, mat, F, V, f1, f2, j1, j2, gaussian_quadrature_order, quadrature_subdivision,
                vertex_map_inverse
            );

            if (is_singular) {
#pragma omp critical
                for (int i = 0; i < 3; i++) B.middleRows<3>(3 * j1[i]) += mat.middleRows<3>(3 * i).cast<ComplexType>();
            }
        }
    }
}

void compute_elastodynamic_B_angular_kernel_galerkin_singular(
    ComplexType *B_buffer, const ScalarType *V_buffer, const IntType *F_buffer, const ScalarType *N_buffer,
    Eigen::Index num_vertices, Eigen::Index num_faces, SingularScalarType c1, SingularScalarType c2,
    std::complex<SingularScalarType> s, unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision,
    const IntType *vertex_map_inverse
) {
    const auto V = Eigen::Map<const MatrixX3s>(V_buffer, num_vertices, (Eigen::Index)3);
    const auto F = Eigen::Map<const MatrixX3i>(F_buffer, num_faces, (Eigen::Index)3);
    const auto N = Eigen::Map<const MatrixX3s>(N_buffer, num_faces, (Eigen::Index)3);

    auto ind_3d = [](auto *tensor, int _dim1, int _dim2, int _dim3, int _ind1, int _ind2, int _ind3) -> auto & {
        return tensor[_dim3 * _dim2 * _ind1 + _dim3 * _ind2 + _ind3]; // Row major
        // return tensor[_dim1*_dim2*_ind3 + _dim1*_ind2 + _ind1]; //Column major
    };

    const auto b_hat_ijk = [&](const auto &r_hat, auto r, const auto &n, int i, int j,
                               int k) -> std::complex<decltype(r)> {
        constexpr auto kronecker_delta = [](int i, int j) -> SingularScalarType { return (i == j) ? 1.0 : 0.0; };
        const SingularComplexType exp_neg_rs_c1 = std::exp(-r * s / c1);
        const SingularComplexType exp_neg_rs_c2 = std::exp(-r * s / c2);

        return -(SingularScalarType)(1. / (4. * M_PI)) *
               ((r_hat[i] * r_hat[k] * n[j] + r_hat[j] * r_hat[k] * n[i] + kronecker_delta(j, k) * n[i] +
                 kronecker_delta(i, k) * n[j]) /
                    (SingularScalarType)2. *
                    (((s * r / c2 + (SingularScalarType)1.) / (s * s)) * exp_neg_rs_c2 -
                     ((s * r / c1 + (SingularScalarType)1.) / (s * s)) * exp_neg_rs_c1) /
                    r -
                (kronecker_delta(j, k) * n(i) + kronecker_delta(i, k) * n(j)) / ((SingularScalarType)2. * s) *
                    (exp_neg_rs_c2 / c2 - exp_neg_rs_c1 / c1) -
                (kronecker_delta(i, j) * n(k)) / (c2 * s) * exp_neg_rs_c2);
    };

    const auto b_hat = [&](const auto &y, const auto &x, const auto &n) {
        const auto r_vec = y - x;
        const auto r_hat = r_vec.stableNormalized().eval();
        const auto r = r_vec.norm();
        Eigen::Matrix<SingularComplexType, 3, 9> mat;
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

#pragma omp parallel for
    for (int f1 = 0; f1 < num_faces; f1++) {
        for (int f2 = 0; f2 < num_faces; f2++) {
            const Eigen::Matrix<SingularScalarType, 1, 3> n = N.row(f2).cast<SingularScalarType>();
            const auto integrand = [&](const auto &x, const auto &y, const auto &Lx, const auto &Ly) {
                auto _b_hat = b_hat(y, x, n);
                Eigen::Matrix<SingularComplexType, 9, 9, Eigen::RowMajor> res;
                for (int i = 0; i < 3; i++) res.middleRows<3>(3 * i) = Lx(i) * _b_hat;
                return res;
            };

            Eigen::Matrix<SingularComplexType, 9, 9, Eigen::RowMajor> mat;
            Eigen::Index j1[3], j2[3];
            bool is_singular = integrate_galerkin_singular<SingularScalarType>(
                integrand, mat, F, V, f1, f2, j1, j2, gaussian_quadrature_order, quadrature_subdivision,
                vertex_map_inverse
            );

            if (is_singular) {
#pragma omp critical
                for (int k = 0; k < 3; k++) {
                    for (int i = 0; i < 3; i++) {
                        for (int p = 0; p < 3; p++) {
                            for (int q = 0; q < 3; q++) {
                                ind_3d(B_buffer, num_vertices * 3, 3, 3, 3 * j1[i] + p, q, k) +=
                                    (ComplexType)mat(3 * i + p, 3 * k + q);
                            }
                        }
                    }
                }
            }
        }
    }
}
