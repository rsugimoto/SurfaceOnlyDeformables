#include "elastics_bem_matrix_collection.hpp"
#include <iostream>

#include "integrator_collocation.hpp"
#include "integrator_galerkin.hpp"
#include "matrix_io.hpp"

#include <igl/barycentric_coordinates.h>
#include <igl/doublearea.h>
#include <igl/per_face_normals.h>
#include <igl/unique_edge_map.h>

ElasticsBEMMatrixCollection::ElasticsBEMMatrixCollection()
    : PhysicsBaseMatrixCollection(), use_galerkin(false), update_local_frame(true), compression_ratio(0.0) {}

bool ElasticsBEMMatrixCollection::init(const nlohmann::json &config) {
    if (!PhysicsBaseMatrixCollection::init(config)) return false;

    nu = config["material"]["nu"].get<ScalarType>();
    mu = config["material"]["mu"].get<ScalarType>();

    if (config["simulation"].contains("compression_ratio"))
        compression_ratio = config["simulation"]["compression_ratio"].get<ScalarType>();
    use_compressed_matrices = compression_ratio != 0.0;

    if (use_compressed_matrices) {
        for (int i = 0; i < num_vertices * 3; i++) {
            compression_permutation.indices()(i) = (i / 3) + (i % 3) * num_vertices;
        }
    }

    if (config["simulation"].contains("use_galerkin")) use_galerkin = config["simulation"]["use_galerkin"].get<bool>();

    if (config["simulation"].contains("update_local_frame"))
        update_local_frame = config["simulation"]["update_local_frame"].get<bool>();

    gaussian_quadrature_order = config["simulation"]["gaussian_quadrature_order"].get<size_t>();
    quadrature_subdivision = config["simulation"]["quadrature_subdivision"].get<size_t>();

    return true;
}

MatrixXs ElasticsBEMMatrixCollection::compute_elastostatic_G_matrix_collocation(const MatrixX3s &init_V) {
    std::cout << "compute elastostatic G matrix collocation" << std::endl;

    constexpr auto kronecker_delta = [](int i, int j) -> ScalarType { return (i == j) ? 1.0 : 0.0; };

    const auto r_u_star_ij = [&](const auto &r_hat, auto r, int i, int j) -> decltype(r) {
        return (1. / (16. * M_PI * (1. - nu) * mu)) * (kronecker_delta(i, j) * (3. - 4. * nu) + r_hat(i) * r_hat(j));
    };

    auto u_star = [&](const auto &y, const auto &x) -> Matrix3s {
        Matrix3s mat;
        auto r_vec = y - x;
        RowVector3s r_hat = r_vec.normalized();
        ScalarType r = r_vec.norm();
        for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) {
                mat(i, j) = (1. / (16. * M_PI * (1. - nu) * mu)) *
                            (kronecker_delta(i, j) * (3. - 4. * nu) + r_hat(i) * r_hat(j)) / r;
            }
        }
        return mat;
    };

    MatrixXs G = MatrixXs::Zero(init_V.rows() * 3, init_V.rows() * 3);

#pragma omp parallel for
    for (Eigen::Index i = 0; i < init_V.rows(); i++) {
        auto x = init_V.row(i);
        for (Eigen::Index f_index = 0; f_index < F.rows(); f_index++) {
            const Eigen::Index j1 = F(f_index, 0), j2 = F(f_index, 1), j3 = F(f_index, 2);
            bool has_singular_point = true;
            int singular_vert_index;
            if (i == j1 || (enable_traction_discontinuity && vertex_map_inverse[i] == vertex_map_inverse[j1]))
                singular_vert_index = 0;
            else if (i == j2 || (enable_traction_discontinuity && vertex_map_inverse[i] == vertex_map_inverse[j2]))
                singular_vert_index = 1;
            else if (i == j3 || (enable_traction_discontinuity && vertex_map_inverse[i] == vertex_map_inverse[j3]))
                singular_vert_index = 2;
            else
                has_singular_point = false;

            if (!has_singular_point) {
                const auto y1 = V.row(j1), y2 = V.row(j2), y3 = V.row(j3);
                const Eigen::Matrix<ScalarType, 9, 3> mat = integrate_collocation_nonsingular(
                    [&](const auto &y) {
                        RowVector3s L;
                        igl::barycentric_coordinates(y, y1, y2, y3, L);
                        L = L.cwiseMin(1.0).cwiseMax(0.0);
                        const auto _u_star = u_star(y, x);
                        Eigen::Matrix<ScalarType, 9, 3> res;
                        for (int j = 0; j < 3; j++) res.middleRows<3>(3 * j) = _u_star * L(j);
                        return res;
                    },
                    y1, y2, y3, gaussian_quadrature_order, quadrature_subdivision
                );
                G.block<3, 3>(3 * i, 3 * j1) += mat.block<3, 3>(0, 0);
                G.block<3, 3>(3 * i, 3 * j2) += mat.block<3, 3>(3, 0);
                G.block<3, 3>(3 * i, 3 * j3) += mat.block<3, 3>(6, 0);
            } else {
                const Eigen::Index j1 = F(f_index, singular_vert_index), j2 = F(f_index, (singular_vert_index + 1) % 3),
                                   j3 = F(f_index, (singular_vert_index + 2) % 3);
                const auto y1 = V.row(j1), y2 = V.row(j2), y3 = V.row(j3);

                integrate_collocation_weakly_singular<ScalarType>(
                    [&](auto func) { // update loop
                        for (int p = 0; p < 3; p++) {
                            for (int q = 0; q <= p; q++) {
                                Vector3s elems = func(Eigen::Vector2i(p, q));
                                G(3 * i + p, 3 * j1 + q) += elems(0);
                                G(3 * i + p, 3 * j2 + q) += elems(1);
                                G(3 * i + p, 3 * j3 + q) += elems(2);
                                if (p != q) {
                                    G(3 * i + q, 3 * j1 + p) += elems(0);
                                    G(3 * i + q, 3 * j2 + p) += elems(1);
                                    G(3 * i + q, 3 * j3 + p) += elems(2);
                                }
                            }
                        }
                    },
                    [&](const auto &y, const auto &r_hat, const auto &r,
                        const auto &indices) -> Vector3s { // integrand
                        RowVector3s L;
                        igl::barycentric_coordinates(y, y1, y2, y3, L);
                        L = L.cwiseMin(1.0).cwiseMax(0.0);
                        return r_u_star_ij(r_hat, r, indices[0], indices[1]) * L;
                    },
                    y1, y2, y3, gaussian_quadrature_order, quadrature_subdivision
                );
            }
        }
    }

    return G;
}

MatrixXs
ElasticsBEMMatrixCollection::compute_elastostatic_H_matrix_collocation(const MatrixX3s &init_V, const MatrixX3s &N) {
    std::cout << "compute elastostatic H matrix collocation" << std::endl;

    constexpr auto kronecker_delta = [](int i, int j) -> ScalarType { return (i == j) ? 1.0 : 0.0; };

    auto p_star_ij = [&](const auto &r_hat, auto r, const auto &n, int i, int j) -> ScalarType {
        return ((1. - 2. * nu) / (8. * M_PI * (1. - nu))) *
               ((r_hat(i) * n(j) - r_hat(j) * n(i)) -
                (kronecker_delta(i, j) + (1. / (1. - 2 * nu)) * (3. * r_hat(i) * r_hat(j))) * (r_hat.dot(n))) /
               (r * r);
    };
    const auto p_star_singular_ij = [&](const auto &r_hat, auto r, const auto &n, int i, int j) -> ScalarType {
        return ((1. - 2. * nu) / (8. * M_PI * (1. - nu))) * (r_hat[i] * n[j] - r_hat[j] * n[i]) / (r * r);
    };
    auto p_star = [&](const auto &y, const auto &x, const auto &n) -> Matrix3s {
        Matrix3s mat;
        auto r_vec = y - x;
        RowVector3s r_hat = r_vec.normalized();
        ScalarType r = r_vec.norm();
        for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) {
                ScalarType elem =
                    ((1. - 2. * nu) / (8. * M_PI * (1. - nu))) *
                    ((r_hat(i) * n(j) - r_hat(j) * n(i)) -
                     (kronecker_delta(i, j) + (1. / (1. - 2 * nu)) * (3. * r_hat(i) * r_hat(j))) * (r_hat.dot(n))) /
                    (r * r);
                mat(i, j) = elem;
            }
        }
        return mat;
    };

    MatrixXs H = MatrixXs::Zero(init_V.rows() * 3, init_V.rows() * 3);
#pragma omp parallel for
    for (Eigen::Index i = 0; i < init_V.rows(); i++) {
        auto x = init_V.row(i);
        for (Eigen::Index f_index = 0; f_index < F.rows(); f_index++) {
            const Eigen::Index j1 = F(f_index, 0), j2 = F(f_index, 1), j3 = F(f_index, 2);
            bool has_singular_point = true;
            int singular_vert_index;
            if (i == j1 || (enable_traction_discontinuity && vertex_map_inverse[i] == vertex_map_inverse[j1]))
                singular_vert_index = 0;
            else if (i == j2 || (enable_traction_discontinuity && vertex_map_inverse[i] == vertex_map_inverse[j2]))
                singular_vert_index = 1;
            else if (i == j3 || (enable_traction_discontinuity && vertex_map_inverse[i] == vertex_map_inverse[j3]))
                singular_vert_index = 2;
            else
                has_singular_point = false;

            const auto n = N.row(f_index);
            if (!has_singular_point) {
                const auto y1 = V.row(j1), y2 = V.row(j2), y3 = V.row(j3);
                const Eigen::Matrix<ScalarType, 9, 3> mat = integrate_collocation_nonsingular(
                    [&](const auto &y) {
                        RowVector3s L;
                        igl::barycentric_coordinates(y, y1, y2, y3, L);
                        L = L.cwiseMin(1.0).cwiseMax(0.0);
                        const auto _p_star = p_star(y, x, n);
                        Eigen::Matrix<ScalarType, 9, 3> res;
                        for (int j = 0; j < 3; j++) res.middleRows<3>(3 * j) = _p_star * L(j);
                        return res;
                    },
                    y1, y2, y3, gaussian_quadrature_order, quadrature_subdivision
                );
                H.block<3, 3>(3 * i, 3 * j1) += mat.block<3, 3>(0, 0);
                H.block<3, 3>(3 * i, 3 * j2) += mat.block<3, 3>(3, 0);
                H.block<3, 3>(3 * i, 3 * j3) += mat.block<3, 3>(6, 0);
            } else {
                const Eigen::Index j1 = F(f_index, singular_vert_index), j2 = F(f_index, (singular_vert_index + 1) % 3),
                                   j3 = F(f_index, (singular_vert_index + 2) % 3);
                const auto y1 = V.row(j1), y2 = V.row(j2), y3 = V.row(j3);

                integrate_collocation_strongly_singular<ScalarType>(
                    [&](auto func) { // update loop
                        for (int p = 0; p < 3; p++) {
                            for (int q = 0; q < 3; q++) {
                                const Vector3s elems = func(Eigen::Vector2i(p, q));
                                H(3 * i + p, 3 * j1 + q) += elems(0);
                                H(3 * i + p, 3 * j2 + q) += elems(1);
                                H(3 * i + p, 3 * j3 + q) += elems(2);
                            }
                        }
                    },
                    [&](const auto &r_hat, const auto &r,
                        const auto &indices) { // original integrand
                        return p_star_ij(r_hat, r, n, indices[0], indices[1]);
                    },
                    [&](const auto &r_hat, const auto &r,
                        const auto &indices) { // cancelling integrand
                        return p_star_singular_ij(r_hat, r, n, indices[0], indices[1]);
                    },
                    y1, y2, y3, n, gaussian_quadrature_order, quadrature_subdivision
                );
            }
        }
    }

#pragma omp parallel for
    for (Eigen::Index i = 0; i < init_V.rows(); i++) {
        H.block<3, 3>(3 * i, 3 * i).setZero();
        for (Eigen::Index j = 0; j < init_V.rows(); j++) {
            if (i == j) continue;
            H.block<3, 3>(3 * i, 3 * i) -= H.block<3, 3>(3 * i, 3 * j);
        }
    }

    return H;
}

MatrixXs ElasticsBEMMatrixCollection::compute_elastostatic_G_matrix_galerkin(const MatrixX3s &init_V) {
    std::cout << "compute elastostatic G matrix galerkin" << std::endl;

    constexpr auto kronecker_delta = [](int i, int j) -> ScalarType { return (i == j) ? 1.0 : 0.0; };

    auto u_star = [&](const auto &y, const auto &x) -> Matrix3s {
        Matrix3s mat;
        auto r_vec = y - x;
        RowVector3s r_hat = r_vec.normalized();
        ScalarType r = r_vec.norm();
        for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) {
                mat(i, j) = (1. / (16. * M_PI * (1. - nu) * mu)) *
                            (kronecker_delta(i, j) * (3. - 4. * nu) + r_hat(i) * r_hat(j)) / r;
            }
        }
        return mat;
    };

    MatrixXs G = MatrixXs::Zero(init_V.rows() * 3, init_V.rows() * 3);
#pragma omp parallel for
    for (int f1 = 0; f1 < F.rows(); f1++) {
        for (int f2 = 0; f2 < F.rows(); f2++) {
            const auto integrand = [&](const auto &x, const auto &y, const auto &Lx, const auto &Ly) {
                auto _u_star = u_star(y, x);
                Eigen::Matrix<ScalarType, 9, 9, Eigen::RowMajor> res;
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) { res.block<3, 3>(3 * i, 3 * j) = Lx(i) * Ly(j) * _u_star; }
                }
                return res;
            };

            Eigen::Matrix<ScalarType, 9, 9, Eigen::RowMajor> mat;
            Eigen::Index j1[3], j2[3];
            bool is_singular = integrate_galerkin_singular<ScalarType>(
                integrand, mat, F, V, f1, f2, j1, j2, gaussian_quadrature_order, quadrature_subdivision,
                vertex_map_inverse.data()
            );

            // non-singular
            if (!is_singular) {
                j1[0] = F(f1, 0), j1[1] = F(f1, 1), j1[2] = F(f1, 2);
                j2[0] = F(f2, 0), j2[1] = F(f2, 1), j2[2] = F(f2, 2);

                mat = integrate_galerkin_nonsingular(
                    integrand, V.row(j1[0]), V.row(j1[1]), V.row(j1[2]), V.row(j2[0]), V.row(j2[1]), V.row(j2[2]),
                    gaussian_quadrature_order, quadrature_subdivision
                );
            }

#pragma omp critical
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) { G.block<3, 3>(3 * j1[i], 3 * j2[j]) += mat.block<3, 3>(3 * i, 3 * j); }
            }
        }
    }

    return G;
}

MatrixXs
ElasticsBEMMatrixCollection::compute_elastostatic_H_matrix_galerkin(const MatrixX3s &init_V, const MatrixX3s &N) {
    std::cout << "compute elastostatic H matrix galerkin" << std::endl;

    constexpr auto kronecker_delta = [](int i, int j) -> ScalarType { return (i == j) ? 1.0 : 0.0; };

    auto p_star = [&](const auto &y, const auto &x, const auto &n) -> Matrix3s {
        Matrix3s mat;
        auto r_vec = y - x;
        RowVector3s r_hat = r_vec.normalized();
        ScalarType r = r_vec.norm();
        for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) {
                ScalarType elem =
                    ((1. - 2. * nu) / (8. * M_PI * (1. - nu))) *
                    ((r_hat(i) * n(j) - r_hat(j) * n(i)) -
                     (kronecker_delta(i, j) + (1. / (1. - 2 * nu)) * (3. * r_hat(i) * r_hat(j))) * (r_hat.dot(n))) /
                    (r * r);
                mat(i, j) = elem;
            }
        }
        return mat;
    };

    MatrixXs H = MatrixXs::Zero(init_V.rows() * 3, init_V.rows() * 3);

#pragma omp parallel for
    for (int f1 = 0; f1 < F.rows(); f1++) {
        for (int f2 = 0; f2 < F.rows(); f2++) {
            const auto n = N.row(f2);
            const auto integrand = [&](const auto &x, const auto &y, const auto &Lx, const auto &Ly) {
                auto _p_star = p_star(y, x, n);
                Eigen::Matrix<ScalarType, 9, 9, Eigen::RowMajor> res;
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) { res.block<3, 3>(3 * i, 3 * j) = Lx(i) * Ly(j) * _p_star; }
                }
                return res;
            };

            Eigen::Matrix<ScalarType, 9, 9, Eigen::RowMajor> mat;
            Eigen::Index j1[3], j2[3];
            bool is_singular = integrate_galerkin_singular<ScalarType>(
                integrand, mat, F, V, f1, f2, j1, j2, gaussian_quadrature_order, quadrature_subdivision,
                vertex_map_inverse.data()
            );

            // non-singular
            if (!is_singular) {
                j1[0] = F(f1, 0), j1[1] = F(f1, 1), j1[2] = F(f1, 2);
                j2[0] = F(f2, 0), j2[1] = F(f2, 1), j2[2] = F(f2, 2);

                mat = integrate_galerkin_nonsingular(
                    integrand, V.row(j1[0]), V.row(j1[1]), V.row(j1[2]), V.row(j2[0]), V.row(j2[1]), V.row(j2[2]),
                    gaussian_quadrature_order, quadrature_subdivision
                );
            }

#pragma omp critical
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) { H.block<3, 3>(3 * j1[i], 3 * j2[j]) += mat.block<3, 3>(3 * i, 3 * j); }
            }
        }
    }

    SparseMatrix C = compute_C_matrix_galerkin(N);
    H += C;

#pragma omp parallel for
    for (Eigen::Index i = 0; i < init_V.rows(); i++) {
        H.block<3, 3>(3 * i, 3 * i).setZero();
        for (Eigen::Index j = 0; j < init_V.rows(); j++) {
            if (i == j) continue;
            H.block<3, 3>(3 * i, 3 * i) -= H.block<3, 3>(3 * i, 3 * j);
        }
    }

    return H;
}

MatrixX3s ElasticsBEMMatrixCollection::compute_C_matrix_collocation(const MatrixX3s &N) {
    MatrixX3s C = MatrixX3s::Zero(num_vertices * 3, 3);

    const MatrixX3s *_V = &V;
    const MatrixX3i *_F = &F;
    const MatrixX3s *_N = &N;

    MatrixX3s original_N;
    if (enable_traction_discontinuity) {
        _V = &original_V;
        _F = &original_F;
        _N = &original_N;
        igl::per_face_normals(original_V, original_F, original_N);
    }

    constexpr auto sgn = [](ScalarType x) -> ScalarType {
        if (std::abs(x) < Eigen::NumTraits<ScalarType>::dummy_precision()) return 0.0;
        if (x > 0.0) return 1.0;
        return -1.0;
    };
    VectorXi EMAP;
    Eigen::MatrixXi E, uE;
    std::vector<std::vector<Eigen::Index>> uE2E;
    igl::unique_edge_map(*_F, E, uE, EMAP, uE2E);

    for (Eigen::Index i = 0; i < num_vertices; i++) C.middleRows<3>(3 * i) += (ScalarType)0.5 * Matrix3s::Identity();

    for (Eigen::Index e = 0; e < uE.rows(); e++) {
        Eigen::Index e1 = uE2E[e][0], e2 = uE2E[e][1];
        Eigen::Index f1 = e1 % _F->rows(), f2 = e2 % _F->rows();
        const Vector3s n1 = _N->row(f1), n2 = _N->row(f2);
        const Vector3s r = (_V->row(E(e1, 0)) - _V->row(E(e1, 1))).stableNormalized();

        const ScalarType weight = (1. / (4. * M_PI)) * sgn((n1.cross(n2)).dot(r)) *
                                  std::acos(std::clamp(n1.dot(n2), (ScalarType)-1.0, (ScalarType)1.0));
        C.middleRows<3>(3 * E(e1, 0)) += weight * Matrix3s::Identity();
        C.middleRows<3>(3 * E(e1, 1)) += weight * Matrix3s::Identity();
    }

    for (Eigen::Index f = 0; f < _F->rows(); f++) {
        const Vector3s n = _N->row(f);
        for (int i = 0; i < 3; i++) {
            const Vector3s r1 = (_V->row((*_F)(f, (i + 1) % 3)) - _V->row((*_F)(f, i))).stableNormalized();
            const Vector3s r2 = (_V->row((*_F)(f, (i + 2) % 3)) - _V->row((*_F)(f, i))).stableNormalized();
            C.middleRows<3>(3 * (*_F)(f, i)) -= (1. / (8. * M_PI * (1. - nu))) * ((r2 - r1).cross(n)) * (n.transpose());
        }
    }

    if (enable_traction_discontinuity) {
        // copy appropriate C matrices to the verts added
        MatrixX3s _C = C;
        for (Eigen::Index v = 0; v < V.rows(); v++) {
            _C.middleRows<3>(3 * v) = C.middleRows<3>(3 * vertex_map_inverse(v));
        }
        C.swap(_C);
    }

    return C;
}

SparseMatrix ElasticsBEMMatrixCollection::compute_C_matrix_galerkin(const MatrixX3s &N) {
    SparseMatrix C(num_vertices * 3, num_vertices * 3);

    VectorXs A;
    igl::doublearea(V, F, A);

    std::vector<Eigen::Triplet<ScalarType>> elems;
    elems.reserve(27 * F.rows());
    for (int f = 0; f < F.rows(); f++) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                elems.emplace_back(3 * F(f, i), 3 * F(f, j), i == j ? A(f) / 24. : A(f) / 48.);
                elems.emplace_back(3 * F(f, i) + 1, 3 * F(f, j) + 1, i == j ? A(f) / 24. : A(f) / 48.);
                elems.emplace_back(3 * F(f, i) + 2, 3 * F(f, j) + 2, i == j ? A(f) / 24. : A(f) / 48.);
            }
        }
    }
    C.setFromTriplets(elems.begin(), elems.end());

    return C;
}