#ifndef __INTEGRATOR_HPP__
#define __INTEGRATOR_HPP__

#include <Eigen/Core>
#include <Eigen/Dense>

#ifndef __device__
#define __device__
#endif // !__device__

#ifndef __host__
#define __host__
#endif // !__host__

template <class Func, class Scalar>
__device__ __host__ auto integrate_subdivision_1d(Func func, Scalar a, Scalar b, unsigned int subdivision = 10) {
    auto result = func(a, a + (b - a) / subdivision);
    for (unsigned int i = 1; i < subdivision; i++) {
        result += func(a + (b - a) * i / subdivision, a + (b - a) * (i + 1) / subdivision);
    }
    return result;
}

template <class Func, class Derived1, class Derived2, class Derived3>
__device__ __host__ auto integrate_subdivision_triangle(
    Func func, const Eigen::MatrixBase<Derived1> &x0, const Eigen::MatrixBase<Derived2> &x1,
    const Eigen::MatrixBase<Derived3> &x2, unsigned int subdivision
) {
    auto result = func(x0, (x0 + (x1 - x0) / subdivision).eval(), (x0 + (x2 - x0) / subdivision).eval());
    for (unsigned int i = 0; i < subdivision; i++) {
        for (unsigned int j = 0; j < subdivision; j++) {
            if ((!(i == 0 && j == 0)) && i + j < subdivision) {
                result += func(
                    (x0 + (i * (x1 - x0) + j * (x2 - x0)) / subdivision).eval(),
                    (x0 + (x1 - x0) / subdivision + (i * (x1 - x0) + j * (x2 - x0)) / subdivision).eval(),
                    (x0 + (x2 - x0) / subdivision + (i * (x1 - x0) + j * (x2 - x0)) / subdivision).eval()
                );
            }
            if (i + j < subdivision - 1) {
                result += func(
                    (x0 + (x1 - x0) / subdivision + (x2 - x0) / subdivision +
                     (i * (x1 - x0) + j * (x2 - x0)) / subdivision)
                        .eval(),
                    (x0 + (x1 - x0) / subdivision + (i * (x1 - x0) + j * (x2 - x0)) / subdivision).eval(),
                    (x0 + (x2 - x0) / subdivision + (i * (x1 - x0) + j * (x2 - x0)) / subdivision).eval()
                );
            }
        }
    }
    return result;
}

// Cowper, G. R. “Gaussian quadrature formulas for triangles”. International Journal for Numerical Methods in
// Engineering 7.3 (1973), 405–408. https://doi.org/10.1002/nme.1620070316
template <class Func, class Derived1, class Derived2, class Derived3>
__device__ __host__ auto gaussian_quadrature_triangle(
    Func func, const Eigen::MatrixBase<Derived1> &x0, const Eigen::MatrixBase<Derived2> &x1,
    const Eigen::MatrixBase<Derived3> &x2, int order
) {
    using ScalarType = typename Eigen::internal::traits<Derived1>::Scalar;
    constexpr ScalarType COEFFS_1[][4] = {{1. / 3., 1. / 3., 1. / 3., 1.}};
    constexpr ScalarType COEFFS_2[][4] = {
        {0., 1. / 2., 1. / 2., 1. / 3.}, {1. / 2., 0., 1. / 2., 1. / 3.}, {1. / 2., 1. / 2., 0., 1. / 3.}};
    constexpr ScalarType COEFFS_3[][4] = {
        {1. / 3., 1. / 3., 1. / 3., -27. / 48.},
        {11. / 15., 2. / 15., 2. / 15., 25. / 48.},
        {2. / 15., 11. / 15., 2. / 15., 25. / 48.},
        {2. / 15., 2. / 15., 11. / 15., 25. / 48.}};

    constexpr ScalarType a4 = 0.445948490915965;
    constexpr ScalarType b4 = 0.111690794839005 * 2.;
    constexpr ScalarType c4 = 0.091576213509771;
    constexpr ScalarType d4 = 0.054975871827661 * 2.;
    constexpr ScalarType COEFFS_4[][4] = {{1. - 2. * a4, a4, a4, b4}, {a4, 1. - 2. * a4, a4, b4},
                                          {a4, a4, 1. - 2. * a4, b4}, {1. - 2. * c4, c4, c4, d4},
                                          {c4, 1. - 2. * c4, c4, d4}, {c4, c4, 1. - 2. * c4, d4}};
    constexpr ScalarType a5 = 0.470142064105115;
    constexpr ScalarType b5 = 0.0661970763942530 * 2.;
    constexpr ScalarType c5 = 0.101286507323456;
    constexpr ScalarType d5 = 0.0629695902724135 * 2.;
    constexpr ScalarType COEFFS_5[][4] = {
        {1. / 3., 1. / 3., 1. / 3., 9. / 40.},
        {1. - 2. * a5, a5, a5, b5},
        {a5, 1. - 2. * a5, a5, b5},
        {a5, a5, 1. - 2. * a5, b5},
        {1. - 2. * c5, c5, c5, d5},
        {c5, 1. - 2. * c5, c5, d5},
        {c5, c5, 1. - 2. * c5, d5}};

    decltype(func(x0)) result = decltype(func(x0))::Zero();
    const auto evaluate = [&](auto &COEFFS) {
        for (auto &COEFF : COEFFS) {
            result += func((x0 * COEFF[0] + x1 * COEFF[1] + x2 * COEFF[2]).eval()) * COEFF[3];
        }
    };

    switch (order) {
    case 1: evaluate(COEFFS_1); break;
    case 2: evaluate(COEFFS_2); break;
    case 3: evaluate(COEFFS_3); break;
    case 4: evaluate(COEFFS_4); break;
    case 5: evaluate(COEFFS_5); break;
    }
    const auto v0 = (x1 - x0).eval();
    const auto v1 = (x2 - x0).eval();
    ScalarType double_area = 0.0;
    if constexpr (Eigen::MatrixBase<Derived1>::SizeAtCompileTime == 2)
        double_area = std::abs(v0(0) * v1(1) - v0(1) * v1(0));
    else if constexpr (Eigen::MatrixBase<Derived1>::SizeAtCompileTime == 3)
        double_area = (v0.cross(v1)).norm();
    result *= double_area / 2.0;
    return result;
}

template <class Func, typename Scalar>
__device__ __host__ auto gaussian_quadrature_1d(Func func, Scalar a, Scalar b, int order = 5) {
    constexpr Scalar COEFFS_1[][2] = {{0, 2}};
    constexpr Scalar COEFFS_2[][2] = {{-0.5773502691896257645092, 1}, {0.5773502691896257645092, 1}};
    constexpr Scalar COEFFS_3[][2] = {
        {-0.7745966692414833770359, 0.5555555555555555555556},
        {0, 0.8888888888888888888889},
        {0.7745966692414833770359, 0.555555555555555555556}};
    constexpr Scalar COEFFS_4[][2] = {
        {-0.861136311594052575224, 0.3478548451374538573731},
        {-0.3399810435848562648027, 0.6521451548625461426269},
        {0.3399810435848562648027, 0.6521451548625461426269},
        {0.861136311594052575224, 0.3478548451374538573731}};
    constexpr Scalar COEFFS_5[][2] = {
        {-0.9061798459386639927976, 0.2369268850561890875143},
        {-0.5384693101056830910363, 0.4786286704993664680413},
        {0, 0.5688888888888888888889},
        {0.5384693101056830910363, 0.4786286704993664680413},
        {0.9061798459386639927976, 0.2369268850561890875143}};
    decltype(func(0.0)) result;
    const auto evaluate = [&](auto &COEFFS) {
        result = func((b - a) * COEFFS[0][0] / 2. + (a + b) / 2.) * COEFFS[0][1];
        for (unsigned int i = 1; i < sizeof(COEFFS) / sizeof(COEFFS[0]); i++) {
            result += func((b - a) * COEFFS[i][0] / 2. + (a + b) / 2.) * COEFFS[i][1];
        }
    };
    switch (order) {
    case 1: evaluate(COEFFS_1); break;
    case 2: evaluate(COEFFS_2); break;
    case 3: evaluate(COEFFS_3); break;
    case 4: evaluate(COEFFS_4); break;
    case 5: evaluate(COEFFS_5); break;
    }
    result *= ((b - a) / 2.0);
    return result;
}

#endif //!__INTEGRATOR_HPP__