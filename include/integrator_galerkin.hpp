#ifndef __INTEGRATOR_GALERKIN_HPP__
#define __INTEGRATOR_GALERKIN_HPP__

#include "integrator.hpp"

// Integration of singular and not hyper-singular kernels over two triangles based on
// Sauter S.A., Schwab C. (2010) Generating the Matrix Coefficients. In: Boundary Element Methods. Springer Series in
// Computational Mathematics, vol 39. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-540-68093-2_5

template <class ScalarType, class Func>
__host__ __device__ auto integrate_1D_unit_line(Func func, int order, int subdivision) {
    return integrate_subdivision_1d(
        [&](const ScalarType x_min, const ScalarType x_max) {
            return gaussian_quadrature_1d([&](const ScalarType x) { return func(x); }, x_min, x_max, order);
        },
        (ScalarType)0, (ScalarType)1, subdivision
    );
}

template <class ScalarType, class Func>
__host__ __device__ auto integrate_4D_unit_hypercube(Func func, int order, int subdivision) {
    return integrate_1D_unit_line<ScalarType>(
        [&](const ScalarType x1) {
            return integrate_1D_unit_line<ScalarType>(
                [&](const ScalarType x2) {
                    return integrate_1D_unit_line<ScalarType>(
                        [&](const ScalarType x3) {
                            return integrate_1D_unit_line<ScalarType>(
                                [&](const ScalarType x4) { return func(x1, x2, x3, x4); }, order, subdivision
                            );
                        },
                        order, subdivision
                    );
                },
                order, subdivision
            );
        },
        order, subdivision
    );
}

template <class Func, class Derived>
__host__ __device__ auto integrate_galerkin_face_coincident(
    Func func, const Eigen::MatrixBase<Derived> &x0, const Eigen::MatrixBase<Derived> &x1,
    const Eigen::MatrixBase<Derived> &x2, int order, int subdivision
) {
    using ScalarType = typename Eigen::internal::traits<Derived>::Scalar;
    using Vector2s = Eigen::Matrix<ScalarType, 2, 1>;
    using RowVector3s = Eigen::Matrix<ScalarType, 1, 3>;

    const auto vec0 = (x1 - x0).eval();
    const auto vec1 = (x2 - x1).eval();

    const auto eval_func = [&](const auto &x_hat, const auto &y_hat) {
        return func(
            (x0 + vec0 * x_hat[0] + vec1 * x_hat[1]).eval(),                       // x pos
            (x0 + vec0 * y_hat[0] + vec1 * y_hat[1]).eval(),                       // y pos
            RowVector3s((ScalarType)1. - x_hat[0], x_hat[0] - x_hat[1], x_hat[1]), // x barycentric coord
            RowVector3s((ScalarType)1. - y_hat[0], y_hat[0] - y_hat[1], y_hat[1])  // y barycentric coord
        );
    };

    auto result = integrate_4D_unit_hypercube<ScalarType>(
        [&](const auto eta1, const auto eta2, const auto eta3, const auto xi) {
            Vector2s x_hat, y_hat;
            auto result = decltype(eval_func(x_hat, y_hat))::Zero().eval();
            // 1
            x_hat[0] = xi, x_hat[1] = xi * ((ScalarType)1. - eta1 + eta1 * eta2),
            y_hat[0] = xi * ((ScalarType)1. - eta1 * eta2 * eta3), y_hat[1] = xi * ((ScalarType)1. - eta1);
            result += eval_func(x_hat, y_hat);
            // 2t[
            x_hat[0] = xi * ((ScalarType)1. - eta1 * eta2 * eta3), x_hat[1] = xi * ((ScalarType)1. - eta1),
            y_hat[0] = xi, y_hat[1] = xi * ((ScalarType)1. - eta1 + eta1 * eta2);
            result += eval_func(x_hat, y_hat);
            // 3
            x_hat[0] = xi, x_hat[1] = xi * eta1 * ((ScalarType)1. - eta2 + eta2 * eta3),
            y_hat[0] = xi * ((ScalarType)1. - eta1 * eta2), y_hat[1] = xi * eta1 * ((ScalarType)1. - eta2);
            result += eval_func(x_hat, y_hat);
            // 4
            x_hat[0] = xi * ((ScalarType)1. - eta1 * eta2), x_hat[1] = xi * eta1 * ((ScalarType)1. - eta2),
            y_hat[0] = xi, y_hat[1] = xi * eta1 * ((ScalarType)1. - eta2 + eta2 * eta3);
            result += eval_func(x_hat, y_hat);
            // 5
            x_hat[0] = xi * ((ScalarType)1. - eta1 * eta2 * eta3),
            x_hat[1] = xi * eta1 * ((ScalarType)1. - eta2 * eta3), y_hat[0] = xi,
            y_hat[1] = xi * eta1 * ((ScalarType)1. - eta2);
            result += eval_func(x_hat, y_hat);
            // 6
            x_hat[0] = xi, x_hat[1] = xi * eta1 * ((ScalarType)1. - eta2),
            y_hat[0] = xi * ((ScalarType)1. - eta1 * eta2 * eta3),
            y_hat[1] = xi * eta1 * ((ScalarType)1. - eta2 * eta3);
            result += eval_func(x_hat, y_hat);

            result *= xi * xi * xi * eta1 * eta1 * eta2;
            return result;
        },
        order, subdivision
    );

    ScalarType jacobian = (vec0.cross(vec1)).norm();
    result *= jacobian * jacobian;
    return result;
}

// xy0 and xy1 are shared between two triangles
template <class Func, class Derived>
__host__ __device__ auto integrate_galerkin_edge_adjacent(
    Func func, const Eigen::MatrixBase<Derived> &xy0, const Eigen::MatrixBase<Derived> &xy1,
    const Eigen::MatrixBase<Derived> &x2, const Eigen::MatrixBase<Derived> &y2, int order, int subdivision
) {
    using ScalarType = typename Eigen::internal::traits<Derived>::Scalar;
    using Vector2s = Eigen::Matrix<ScalarType, 2, 1>;
    using RowVector3s = Eigen::Matrix<ScalarType, 1, 3>;

    const auto vec0_xy = (xy1 - xy0).eval();
    const auto vec1_x = (x2 - xy1).eval();
    const auto vec1_y = (y2 - xy1).eval();

    const auto eval_func = [&](const auto &x_hat, const auto &y_hat) {
        return func(
            (xy0 + vec0_xy * x_hat[0] + vec1_x * x_hat[1]).eval(),                 // x pos
            (xy0 + vec0_xy * y_hat[0] + vec1_y * y_hat[1]).eval(),                 // y pos
            RowVector3s((ScalarType)1. - x_hat[0], x_hat[0] - x_hat[1], x_hat[1]), // x barycentric coord
            RowVector3s((ScalarType)1. - y_hat[0], y_hat[0] - y_hat[1], y_hat[1])  // y barycentric coord
        );
    };

    auto result = integrate_4D_unit_hypercube<ScalarType>(
        [&](const auto eta1, const auto eta2, const auto eta3, const auto xi) {
            Vector2s x_hat, y_hat;
            auto result = decltype(eval_func(x_hat, y_hat))::Zero().eval();
            // 1
            x_hat[0] = xi, x_hat[1] = xi * eta1 * eta3, y_hat[0] = xi * ((ScalarType)1. - eta1 * eta2),
            y_hat[1] = xi * eta1 * ((ScalarType)1. - eta2);
            result += eval_func(x_hat, y_hat);
            // 2
            x_hat[0] = xi, x_hat[1] = xi * eta1, y_hat[0] = xi * ((ScalarType)1. - eta1 * eta2 * eta3),
            y_hat[1] = xi * eta1 * eta2 * ((ScalarType)1. - eta3);
            result += eta2 * eval_func(x_hat, y_hat);
            // 3
            x_hat[0] = xi * ((ScalarType)1. - eta1 * eta2), x_hat[1] = xi * eta1 * ((ScalarType)1. - eta2),
            y_hat[0] = xi, y_hat[1] = xi * eta1 * eta2 * eta3;
            result += eta2 * eval_func(x_hat, y_hat);
            // 4
            x_hat[0] = xi * ((ScalarType)1. - eta1 * eta2 * eta3),
            x_hat[1] = xi * eta1 * eta2 * ((ScalarType)1. - eta3), y_hat[0] = xi, y_hat[1] = xi * eta1;
            result += eta2 * eval_func(x_hat, y_hat);
            // 5
            x_hat[0] = xi * ((ScalarType)1. - eta1 * eta2 * eta3),
            x_hat[1] = xi * eta1 * ((ScalarType)1. - eta2 * eta3), y_hat[0] = xi, y_hat[1] = xi * eta1 * eta2;
            result += eta2 * eval_func(x_hat, y_hat);

            result *= xi * xi * xi * eta1 * eta1;
            return result;
        },
        order, subdivision
    );

    ScalarType jacobian_x = (vec0_xy.cross(vec1_x)).norm();
    ScalarType jacobian_y = (vec0_xy.cross(vec1_y)).norm();
    result *= jacobian_x * jacobian_y;
    return result;
}

// xy0 is shared between two triangles
template <class Func, class Derived>
__host__ __device__ auto integrate_galerkin_vertex_adjacent(
    Func func, const Eigen::MatrixBase<Derived> &xy0, const Eigen::MatrixBase<Derived> &x1,
    const Eigen::MatrixBase<Derived> &x2, const Eigen::MatrixBase<Derived> &y1, const Eigen::MatrixBase<Derived> &y2,
    int order, int subdivision
) {
    using ScalarType = typename Eigen::internal::traits<Derived>::Scalar;
    using Vector2s = Eigen::Matrix<ScalarType, 2, 1>;
    using RowVector3s = Eigen::Matrix<ScalarType, 1, 3>;

    const auto vec0_x = (x1 - xy0).eval();
    const auto vec1_x = (x2 - x1).eval();
    const auto vec0_y = (y1 - xy0).eval();
    const auto vec1_y = (y2 - y1).eval();

    const auto eval_func = [&](const auto &x_hat, const auto &y_hat) {
        return func(
            (xy0 + vec0_x * x_hat[0] + vec1_x * x_hat[1]).eval(),                  // x pos
            (xy0 + vec0_y * y_hat[0] + vec1_y * y_hat[1]).eval(),                  // y pos
            RowVector3s((ScalarType)1. - x_hat[0], x_hat[0] - x_hat[1], x_hat[1]), // x barycentric coord
            RowVector3s((ScalarType)1. - y_hat[0], y_hat[0] - y_hat[1], y_hat[1])  // y barycentric coord
        );
    };

    auto result = integrate_4D_unit_hypercube<ScalarType>(
        [&](const auto eta1, const auto eta2, const auto eta3, const auto xi) {
            Vector2s x_hat, y_hat;
            auto result = decltype(eval_func(x_hat, y_hat))::Zero().eval();

            // 1
            x_hat[0] = xi, x_hat[1] = xi * eta1, y_hat[0] = xi * eta2, y_hat[1] = xi * eta2 * eta3;
            result += eval_func(x_hat, y_hat);
            // 2
            x_hat[0] = xi * eta2, x_hat[1] = xi * eta2 * eta3, y_hat[0] = xi, y_hat[1] = xi * eta1;
            result += eval_func(x_hat, y_hat);

            result *= xi * xi * xi * eta2;
            return result;
        },
        order, subdivision
    );

    ScalarType jacobian_x = (vec0_x.cross(vec1_x)).norm();
    ScalarType jacobian_y = (vec0_y.cross(vec1_y)).norm();
    result *= jacobian_x * jacobian_y;
    return result;
}

template <class Func, class Derived>
__host__ __device__ auto integrate_galerkin_nonsingular(
    Func func, const Eigen::MatrixBase<Derived> &x0, const Eigen::MatrixBase<Derived> &x1,
    const Eigen::MatrixBase<Derived> &x2, const Eigen::MatrixBase<Derived> &y0, const Eigen::MatrixBase<Derived> &y1,
    const Eigen::MatrixBase<Derived> &y2, int order, int subdivision
) {
    using ScalarType = typename Eigen::internal::traits<Derived>::Scalar;
    using Vector2s = Eigen::Matrix<ScalarType, 2, 1>;
    using RowVector3s = Eigen::Matrix<ScalarType, 1, 3>;

    const auto vec0_x = (x1 - x0).eval();
    const auto vec1_x = (x2 - x0).eval();
    const auto vec0_y = (y1 - y0).eval();
    const auto vec1_y = (y2 - y0).eval();

    auto result = integrate_subdivision_triangle(
        [&](const auto &_x0, const auto &_x1, const auto &_x2) {
            return gaussian_quadrature_triangle(
                [&](const auto &_x) {
                    return integrate_subdivision_triangle(
                        [&](const auto &_y0, const auto &_y1, const auto &_y2) {
                            return gaussian_quadrature_triangle(
                                [&](const auto &_y) {
                                    return func(
                                        (x0 + _x[0] * vec0_x + _x[1] * vec1_x).eval(), // x pos
                                        (y0 + _y[0] * vec0_y + _y[1] * vec1_y).eval(), // y pos
                                        RowVector3s(1. - _x[0] - _x[1], _x[0], _x[1]), // x barycentric coord
                                        RowVector3s(1. - _y[0] - _y[1], _y[0], _y[1])  // y barycentric coord
                                    );
                                },
                                _y0, _y1, _y2, order
                            );
                        },
                        Vector2s(0., 0.), Vector2s(1., 0.), Vector2s(0., 1.), subdivision
                    );
                },
                _x0, _x1, _x2, order
            );
        },
        Vector2s(0., 0.), Vector2s(1., 0.), Vector2s(0., 1.), subdivision
    );

    ScalarType jacobian_x = (vec0_x.cross(vec1_x)).norm();
    ScalarType jacobian_y = (vec0_y.cross(vec1_y)).norm();
    result *= jacobian_x * jacobian_y;
    return result;
}

template <class SingularScalarType, class Func, class DerivedM, class DerivedF, class DerivedV, typename IntType>
__host__ __device__ bool integrate_galerkin_singular(
    Func integrand, Eigen::MatrixBase<DerivedM> &mat, const Eigen::MatrixBase<DerivedF> &F,
    const Eigen::MatrixBase<DerivedV> &V, int f1, int f2, Eigen::Index j1[3], Eigen::Index j2[3],
    unsigned int gaussian_quadrature_order, unsigned int quadrature_subdivision, const IntType *vertex_map_inverse
) {

    bool is_face_coincident = false, is_edge_adjacent = false, is_vertex_adjacent = false;

    // face coincident
    {
        is_face_coincident = (f1 == f2);
        if (is_face_coincident) {
            j1[0] = j2[0] = F(f1, 0), j1[1] = j2[1] = F(f1, 1), j1[2] = j2[2] = F(f1, 2);
            mat = integrate_galerkin_face_coincident(
                integrand, V.row(j1[0]).template cast<SingularScalarType>().eval(),
                V.row(j1[1]).template cast<SingularScalarType>().eval(),
                V.row(j1[2]).template cast<SingularScalarType>().eval(), gaussian_quadrature_order,
                quadrature_subdivision
            );
        }
    }

    // edge adjacent
    if (!is_face_coincident) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (F(f1, i) == F(f2, j) && F(f1, (i + 1) % 3) == F(f2, (j + 2) % 3) ||
                    (vertex_map_inverse != nullptr && vertex_map_inverse[F(f1, i)] == vertex_map_inverse[F(f2, j)] &&
                     vertex_map_inverse[F(f1, (i + 1) % 3)] == vertex_map_inverse[F(f2, (j + 2) % 3)])) {
                    is_edge_adjacent = true;
                    j1[0] = F(f1, i), j1[1] = F(f1, (i + 1) % 3), j1[2] = F(f1, (i + 2) % 3);
                    j2[0] = F(f2, j), j2[1] = F(f2, (j + 2) % 3), j2[2] = F(f2, (j + 1) % 3);
                    break;
                }
            }
        }
        if (is_edge_adjacent)
            mat = integrate_galerkin_edge_adjacent(
                integrand, V.row(j1[0]).template cast<SingularScalarType>().eval(),
                V.row(j1[1]).template cast<SingularScalarType>().eval(),
                V.row(j1[2]).template cast<SingularScalarType>().eval(),
                V.row(j2[2]).template cast<SingularScalarType>().eval(), gaussian_quadrature_order,
                quadrature_subdivision
            );
    }

    // vertex adjacent
    if (!is_face_coincident && !is_edge_adjacent) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (F(f1, i) == F(f2, j) ||
                    (vertex_map_inverse != nullptr && vertex_map_inverse[F(f1, i)] == vertex_map_inverse[F(f2, j)])) {
                    is_vertex_adjacent = true;
                    j1[0] = F(f1, i), j1[1] = F(f1, (i + 1) % 3), j1[2] = F(f1, (i + 2) % 3);
                    j2[0] = F(f2, j), j2[1] = F(f2, (j + 1) % 3), j2[2] = F(f2, (j + 2) % 3);
                    break;
                }
            }
        }
        if (is_vertex_adjacent)
            mat = integrate_galerkin_vertex_adjacent(
                integrand, V.row(j1[0]).template cast<SingularScalarType>().eval(),
                V.row(j1[1]).template cast<SingularScalarType>().eval(),
                V.row(j1[2]).template cast<SingularScalarType>().eval(),
                V.row(j2[1]).template cast<SingularScalarType>().eval(),
                V.row(j2[2]).template cast<SingularScalarType>().eval(), gaussian_quadrature_order,
                quadrature_subdivision
            );
    }

    return is_vertex_adjacent || is_edge_adjacent || is_face_coincident;
}

#endif //!__INTEGRATOR_GALERKIN_HPP__