#ifndef __INTEGRATOR_COLLOCATION_HPP__
#define __INTEGRATOR_COLLOCATION_HPP__

#include "integrator.hpp"

template <class Func, class Derived1, class Derived2, class Derived3>
__device__ __host__ auto integrate_collocation_nonsingular(
    Func func, const Eigen::MatrixBase<Derived1> &x0, const Eigen::MatrixBase<Derived2> &x1,
    const Eigen::MatrixBase<Derived3> &x2, unsigned int order, unsigned int subdivision
) {
    return integrate_subdivision_triangle(
        [&](const auto &_x0, const auto &_x1, const auto &_x2) {
            return gaussian_quadrature_triangle(func, _x0, _x1, _x2, order);
        },
        x0, x1, x2, subdivision
    );
}

template <typename ScalarType, class Func1, class Func2, class Derived>
__device__ __host__ void integrate_collocation_weakly_singular(
    Func1 loop, Func2 integrand, const Eigen::MatrixBase<Derived> &y1, const Eigen::MatrixBase<Derived> &y2,
    const Eigen::MatrixBase<Derived> &y3, int order, int subdivision
) {
    constexpr auto clamp = [](ScalarType val, ScalarType min, ScalarType max) -> ScalarType {
        return val < min ? min : val > max ? max : val;
    };

    const auto v1 = (y2 - y1).eval(), v2 = (y3 - y1).eval();
    const auto v1_stableNormalized = v1.stableNormalized().eval(), v2_stableNormalized = v2.stableNormalized().eval();
    const auto x_hat = v1_stableNormalized;
    const auto y_hat = ((x_hat.cross(v2_stableNormalized)).cross(x_hat)).stableNormalized().eval();

    const auto v_12 = (v1 + (v1.dot((v1 - v2).stableNormalized())) * ((v2 - v1).stableNormalized())).eval();
    const auto a = v_12.norm();
    const auto alpha = acos(v1_stableNormalized.dot(v2_stableNormalized));
    const auto beta = acos(clamp(v_12.stableNormalized().dot(v1_stableNormalized), (ScalarType)-1.0, (ScalarType)1.0));

    loop([&](auto indices) {
        return integrate_subdivision_1d(
            [&](ScalarType _a, ScalarType _b) {
                return gaussian_quadrature_1d(
                    [&](ScalarType theta) {
                        const auto r_hat = (x_hat * cos(theta) + y_hat * sin(theta)).eval();
                        return integrate_subdivision_1d(
                            [&](ScalarType _a, ScalarType _b) {
                                return gaussian_quadrature_1d(
                                    [&](ScalarType r) {
                                        const auto y = (y1 + r * r_hat).eval();
                                        return integrand(y, r_hat, r, indices);
                                    },
                                    _a, _b, order
                                );
                            },
                            (ScalarType)0.0, a / cos(theta - beta), subdivision
                        );
                    },
                    _a, _b, order
                );
            },
            (ScalarType)0.0, alpha, subdivision
        );
    });
}

// Guiggiani, Massimo and Gigante, A. “A General Algorithm for Multidimensional Cauchy Principal Value Integrals in the
// Boundary Element Method”. Journal of Applied Mechanics 57.4 (Dec. 1990), 906– 915. issn: 0021-8936.
// https://doi.org/10.1115/1.2897660
template <typename ScalarType, class Func1, class Func2, class Func3, class Derived1, class Derived2>
__device__ __host__ void integrate_collocation_strongly_singular(
    Func1 loop, Func2 original_integrand, Func3 cancelling_integrand, const Eigen::MatrixBase<Derived1> &y1,
    const Eigen::MatrixBase<Derived1> &y2, const Eigen::MatrixBase<Derived1> &y3, const Eigen::MatrixBase<Derived2> &n,
    int order, int subdivision
) {
    const auto v1 = (y2 - y1).eval(), v2 = (y3 - y1).eval();
    const auto J = v1.cross(v2).norm();

    loop([&](auto indices) {
        return integrate_subdivision_1d(
            [&](ScalarType _a, ScalarType _b) {
                return gaussian_quadrature_1d(
                    [&](ScalarType theta) {
                        const auto cos_theta = cos(theta), sin_theta = sin(theta);
                        const Eigen::Matrix<ScalarType, 3, 1> L(1.0, 0.0, 0.0);
                        const auto A_vec = (cos_theta * v1 + sin_theta * v2).eval();
                        const auto A_hat = A_vec.stableNormalized().eval();
                        const auto A = A_vec.norm();
                        using MatrixScalarType = decltype(cancelling_integrand(A_hat, A, indices));
                        const auto f_ij =
                            (cancelling_integrand(A_hat, A, indices) * (L * J).template cast<MatrixScalarType>())
                                .eval();

                        const auto rho_bar =
                            (ScalarType)1. / (cos(theta - (ScalarType)(M_PI / 4.)) * sqrt((ScalarType)2.));
                        const auto part1 = integrate_subdivision_1d(
                            [&](ScalarType _a, ScalarType _b) {
                                return gaussian_quadrature_1d(
                                    [&](ScalarType rho) {
                                        const Eigen::Matrix<ScalarType, 3, 1> L(
                                            (ScalarType)1.0 - rho * (cos_theta + sin_theta), rho * cos_theta,
                                            rho * sin_theta
                                        );
                                        const auto r_vec = (rho * A_vec).eval();
                                        const auto r = r_vec.norm();
                                        const auto r_hat = r_vec.stableNormalized().eval();
                                        const auto F_ij = (original_integrand(r_hat, r, indices) *
                                                           (L * J * rho).template cast<MatrixScalarType>())
                                                              .eval();
                                        return (F_ij - f_ij / rho).eval();
                                    },
                                    _a, _b, order
                                );
                            },
                            (ScalarType)0.0, rho_bar, subdivision
                        );
                        const auto part2 = (f_ij * log(rho_bar * A)).eval();
                        return (part1 + part2).eval();
                    },
                    _a, _b, order
                );
            },
            (ScalarType)0.0, (ScalarType)(M_PI / 2.), subdivision
        );
    });
}

#endif //!__INTEGRATOR_COLLOCATION_HPP__