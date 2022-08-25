#include "elastodynamics_drbem_matrix_collection.hpp"

#include <filesystem>
#include <iostream>
#include <sstream>

#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>

#include "integrator_collocation.hpp"
#include "integrator_galerkin.hpp"
#include "matrix_io.hpp"

ElastodynamicsDRBEMMatrixCollection::ElastodynamicsDRBEMMatrixCollection(ScalarType dt)
    : ElasticsBEMMatrixCollection(), dt(dt) {}

ElastodynamicsDRBEMMatrixCollection::~ElastodynamicsDRBEMMatrixCollection() {}

bool ElastodynamicsDRBEMMatrixCollection::init(const nlohmann::json &config) {
    if (!ElasticsBEMMatrixCollection::init(config)) return false;

    std::cout << "type: Elastodynamics DRBEM" << std::endl;

    if (nu >= 0.5) {
        std::cout << "nu>=0.5 is not supported with DRBEM" << std::endl;
        return false;
    }

    if (dt != config["simulation"]["dt"].get<ScalarType>()) {
        std::cout << "dt for simulator and elastodynamics BEM need to be the same." << std::endl;
        return false;
    }

    // Use hashing to store matrices for different configs in separate folders.
    coeffs_folder_path = config["simulation"]["coeffs_folder_path"].get<std::string>() + "/elastodynamicsDRBEM_";
    {
        nlohmann::json config_copy = config;
        const char *configs_ignored[] = {"compression_ratio", "update_local_frame", "dt"};
        for (auto _config : configs_ignored)
            if (config_copy["simulation"].contains(_config))
                config_copy["simulation"].erase(config_copy["simulation"].find(_config));
        std::ostringstream ss;
        ss << std::uppercase << std::hex << std::hash<std::string>()(config_copy.dump());
        coeffs_folder_path += ss.str();

        std::filesystem::create_directories(coeffs_folder_path);

        // save the json config for information
        std::ofstream json_file(coeffs_folder_path + "/config.json");
        json_file << config_copy.dump(4);
    }

    MatrixX3s N;
    igl::per_face_normals(V, F, N);

    MatrixXs _H, G;
    // load/compute H, G, M, B_trans, and B_euler
    if (!Eigen::load_matrix(_H, coeffs_folder_path + "/H.mat") ||
        !Eigen::load_matrix(G, coeffs_folder_path + "/G.mat") ||
        !Eigen::load_matrix(M, coeffs_folder_path + "/M.mat") ||
        !Eigen::load_matrix(B_trans, coeffs_folder_path + "/B_trans.mat") ||
        !Eigen::load_matrix(B_euler, coeffs_folder_path + "/B_euler.mat")) {
        if (use_galerkin) {
            _H = compute_elastostatic_H_matrix_galerkin(V, N);
            G = compute_elastostatic_G_matrix_galerkin(V);
        } else {
            _H = compute_elastostatic_H_matrix_collocation(V, N);
            G = compute_elastostatic_G_matrix_collocation(V);
        }

        M = compute_M_matrix(_H, G);
        B_trans = compute_B_trans_matrix(N);
        TensorXs B_angular = compute_B_angular_tensor(N, cm, B_trans);
        B_euler = compute_B_euler_matrix(N, B_angular);

        Eigen::save_matrix(_H, coeffs_folder_path + "/H.mat");
        Eigen::save_matrix(G, coeffs_folder_path + "/G.mat");
        Eigen::save_matrix(M, coeffs_folder_path + "/M.mat");
        Eigen::save_matrix(B_trans, coeffs_folder_path + "/B_trans.mat");
        Eigen::save_matrix(B_euler, coeffs_folder_path + "/B_euler.mat");
    }

    if (update_local_frame) G += B_trans * translational_acc_mat + B_euler * rotational_acc_mat;
    MatrixXs H = _H + (1. / (dt * dt)) * M;
    _H.resize(0, 0); // free memory

    B_trans_inv.compute(B_trans);
    B_euler_inv.compute(B_euler);

    B_trans_inv_H = B_trans_inv.solve(H);
    B_euler_inv_H = B_euler_inv.solve(H);
    B_trans_inv_G = B_trans_inv.solve(G);
    B_euler_inv_G = B_euler_inv.solve(G);
    B_euler_inv_B_trans = B_euler_inv.solve(B_trans);
    rotation_removal_weights = compute_rotation_removal_weights(N);

    if (!Eigen::load_matrix(
            u_p, coeffs_folder_path + "/u_p" + (update_local_frame ? "" : "no_frame_update_") + ".mat"
        ) ||
        !Eigen::load_matrix(u_b, coeffs_folder_path + "/u_b.mat") ||
        !Eigen::load_matrix(H_inv_B_trans, coeffs_folder_path + "/H_inv_B_trans.mat") ||
        !Eigen::load_matrix(H_inv_B_euler, coeffs_folder_path + "/H_inv_B_euler.mat")) {
        std::cout << "Computing H inverse..." << std::endl;
        Eigen::PartialPivLU<MatrixXs> H_inv(H);
        std::cout << "Finished computing H inverse" << std::endl;

        u_p = H_inv.solve(G);
        u_b = H_inv.solve(M);
        H_inv_B_trans = H_inv.solve(B_trans);
        H_inv_B_euler = H_inv.solve(B_euler);

        Eigen::save_matrix(u_p, coeffs_folder_path + "/u_p" + (update_local_frame ? "" : "no_frame_update_") + ".mat");
        Eigen::save_matrix(u_b, coeffs_folder_path + "/u_b.mat");
        Eigen::save_matrix(H_inv_B_trans, coeffs_folder_path + "/H_inv_B_trans.mat");
        Eigen::save_matrix(H_inv_B_euler, coeffs_folder_path + "/H_inv_B_euler.mat");
    }

    // set u_p_diag
    {
        std::vector<Eigen::Triplet<ScalarType>> elems;
        elems.reserve(3 * 3 * num_vertices);
        for (int v = 0; v < num_vertices; v++) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) { elems.emplace_back(3 * v + i, 3 * v + j, u_p(3 * v + i, 3 * v + j)); }
            }
        }
        u_p_diag.setFromTriplets(elems.begin(), elems.end());
    }

    if (use_compressed_matrices) {
        // u_p
        {
            if (!load_compressed_matrix(
                    u_p_compressed, coeffs_folder_path + "/u_p_" + (update_local_frame ? "" : "no_frame_update_") +
                                        std::to_string(compression_ratio) + ".cpmat"
                )) {
                std::cout << "Compressing u_p matrix..." << std::endl;
                u_p_compressed = CompressedMatrix<>(
                    compression_permutation * u_p * compression_permutation.transpose(), compression_ratio
                );
                save_compressed_matrix(
                    u_p_compressed, coeffs_folder_path + "/u_p_" + (update_local_frame ? "" : "no_frame_update_") +
                                        std::to_string(compression_ratio) + ".cpmat"
                );
                std::cout << "Finished compressing u_p matrix." << std::endl;
            }

            u_p.resize(0, 0); // free memory
        }

        // u_b
        {
            if (!load_compressed_matrix(
                    u_b_compressed, coeffs_folder_path + "/u_b_" + std::to_string(compression_ratio) + ".cpmat"
                )) {
                std::cout << "Compressing u_b..." << std::endl;
                u_b_compressed = CompressedMatrix<>(
                    compression_permutation * u_b * compression_permutation.transpose(), compression_ratio
                );
                save_compressed_matrix(
                    u_b_compressed, coeffs_folder_path + "/u_b_" + std::to_string(compression_ratio) + ".cpmat"
                );
                std::cout << "Finished compressing u_b matrix." << std::endl;
            }

            u_b.resize(0, 0); // free memory
        }

        // M
        {
            if (!load_compressed_matrix(
                    M_compressed, coeffs_folder_path + "/M_" + std::to_string(compression_ratio) + ".cpmat"
                )) {
                std::cout << "Compressing M..." << std::endl;
                M_compressed = CompressedMatrix<>(
                    compression_permutation * M * compression_permutation.transpose(), compression_ratio
                );
                save_compressed_matrix(
                    M_compressed, coeffs_folder_path + "/M_" + std::to_string(compression_ratio) + ".cpmat"
                );
                std::cout << "Finished compressing M matrix." << std::endl;
            }

            M.resize(0, 0); // free memory
        }
    }

    return true;
}

MatrixXs ElastodynamicsDRBEMMatrixCollection::compute_M_matrix(const MatrixXs &P, const MatrixXs &U) {
    std::cout << "compute M matrix" << std::endl;
    assert(U.rows() == U.cols() && U.rows() == num_vertices * 3);
    assert(P.rows() == P.cols() && P.rows() == num_vertices * 3);

    MatrixXs M;
    {
        MatrixX3s N;
        igl::per_vertex_normals(V, F, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, N);

        auto Eta =
            MatrixXs::NullaryExpr(num_vertices * 3, num_vertices * 3, [N, this](Eigen::Index col, Eigen::Index row) {
                constexpr auto kronecker_delta = [](int i, int j) -> ScalarType { return (i == j) ? 1.0 : 0.0; };
                auto r_vec = V.row(col / 3) - V.row(row / 3);
                RowVector3s r_hat = r_vec.stableNormalized();
                ScalarType r = r_vec.norm();
                auto n = N.row(col / 3);
                Eigen::Index i = col % 3, j = row % 3;
                return 1. / (2. * (1. - nu)) *
                       (((1. - 2. * nu) * (1. / 3. + r / 4.) + 1. / 5. + r / 6.) *
                            (r_vec(i) * n(j) + kronecker_delta(i, j) * r_vec.dot(n)) -
                        ((1. - 2. * nu) * (1. / 3. + r / 4.) - 1. / 5. - r / 6.) * r_vec(j) * n(i) -
                        (1. / 12.) * r_vec(i) * r_vec(j) * r_hat.dot(n));
            });

        M = rho * U * Eta;
    }

    {
        auto Psi =
            MatrixXs::NullaryExpr(num_vertices * 3, num_vertices * 3, [this](Eigen::Index col, Eigen::Index row) {
                constexpr auto kronecker_delta = [](int i, int j) -> ScalarType { return (i == j) ? 1.0 : 0.0; };
                auto r_vec = V.row(col / 3) - V.row(row / 3);
                ScalarType r = r_vec.norm();
                Eigen::Index i = col % 3, j = row % 3;
                return 1. / (4. * mu * (1. - nu)) *
                       (kronecker_delta(i, j) *
                            ((3. - 4. * nu) * (r * r / 6. + r * r * r / 12.) + r * r / 10. + r * r * r / 18.) -
                        (2. / 15. + r / 12.) * r_vec(i) * r_vec(j));
            });

        M -= rho * P * Psi;
    }

    {
        MatrixXs F_compressed = MatrixXs::Zero(num_vertices, num_vertices);
#pragma omp parallel for
        for (Eigen::Index i = 0; i < num_vertices; i++) {
            auto y = V.row(i);
            for (Eigen::Index j = 0; j < num_vertices; j++) {
                auto x = V.row(j);
                ScalarType elem = 1 + (y - x).norm();
                F_compressed(i, j) = elem;
            }
        }
        MatrixXs F_inv_compressed;
        if (enable_traction_discontinuity)
            F_inv_compressed = Eigen::CompleteOrthogonalDecomposition<MatrixXs>(F_compressed).pseudoInverse();
        else
            F_inv_compressed = F_compressed.partialPivLu().inverse();
        auto F_inv = MatrixXs::NullaryExpr(
            num_vertices * 3, num_vertices * 3,
            [&F_inv_compressed](Eigen::Index i, Eigen::Index j) {
                return i % 3 == j % 3 ? F_inv_compressed(i / 3, j / 3) : 0.0;
            }
        );

        M *= F_inv;
    }

    return M;
}

MatrixX3s ElastodynamicsDRBEMMatrixCollection::compute_B_trans_matrix(const MatrixX3s &N) {
    std::cout << "Compute B_trans matrix" << std::endl;

    MatrixX3s B = MatrixX3s::Zero(num_vertices * 3, 3);

    constexpr auto kronecker_delta = [](int i, int j) -> ScalarType { return (i == j) ? 1.0 : 0.0; };
    const auto b_star_ij = [&](const auto &r_hat, auto r, const auto &n, int i, int j) {
        return -rho * (1. / (8. * M_PI * mu)) *
               (kronecker_delta(i, j) * r_hat.dot(n) - 1. / (2. * (1. - nu)) * r_hat(j) * n(i));
    };

    const auto b_star = [&](const auto &y, const auto &x, const auto &n) {
        const auto r_vec = y - x;
        const auto r_hat = r_vec.stableNormalized().eval();
        const auto r = r_vec.norm();
        Matrix3s mat;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j <= i; j++) {
                mat(i, j) = b_star_ij(r_hat, r, n, i, j);
                if (i != j) mat(j, i) = mat(i, j);
            }
        }
        return mat;
    };

    if (use_galerkin) {
#pragma omp parallel for
        for (int f1 = 0; f1 < F.rows(); f1++) {
            for (int f2 = 0; f2 < F.rows(); f2++) {
                const auto n = N.row(f2);
                const auto integrand = [&](const auto &x, const auto &y, const auto &Lx, const auto &Ly) {
                    auto _b_star = b_star(y, x, n);
                    Eigen::Matrix<ScalarType, 9, 3, Eigen::RowMajor> res;
                    for (int i = 0; i < 3; i++) res.middleRows<3>(3 * i) = Lx(i) * _b_star;
                    return res;
                };

                Eigen::Matrix<ScalarType, 9, 3, Eigen::RowMajor> mat;
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
                for (int i = 0; i < 3; i++) { B.middleRows<3>(3 * j1[i]) += mat.middleRows<3>(3 * i); }
            }
        }
    } else {
#pragma omp parallel for
        for (Eigen::Index i = 0; i < num_vertices; i++) {
            const auto x = V.row(i);
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
                    const auto n = N.row(f_index);
                    const auto y1 = V.row(j1), y2 = V.row(j2), y3 = V.row(j3);
                    B.middleRows<3>(3 * i) += integrate_collocation_nonsingular(
                        [&](const auto &y) { return b_star(y, x, n); }, y1, y2, y3, gaussian_quadrature_order,
                        quadrature_subdivision
                    );
                } else {
                    const auto n = N.row(f_index);
                    const Eigen::Index j1 = F(f_index, singular_vert_index),
                                       j2 = F(f_index, (singular_vert_index + 1) % 3),
                                       j3 = F(f_index, (singular_vert_index + 2) % 3);
                    const auto y1 = V.row(j1), y2 = V.row(j2), y3 = V.row(j3);

                    integrate_collocation_weakly_singular<ScalarType>(
                        [&](auto func) { // update loop
                            for (int p = 0; p < 3; p++) {
                                for (int q = 0; q <= p; q++) {
                                    ScalarType elem = func(Eigen::Vector2i(p, q));
                                    B(3 * i + p, q) += elem;
                                    if (p != q) B(3 * i + q, p) += elem;
                                }
                            }
                        },
                        [&](const auto &y, const auto &r_hat, const auto &r,
                            const auto &indices) -> ScalarType { // integrand
                            return r * b_star_ij(r_hat, r, n, indices[0], indices[1]);
                        },
                        y1, y2, y3, gaussian_quadrature_order, quadrature_subdivision
                    );
                }
            }
        }
    }

    return B;
}

TensorXs ElastodynamicsDRBEMMatrixCollection::compute_B_angular_tensor(
    const MatrixX3s &N, const RowVector3s &cm, const MatrixX3s &B_trans
) {
    std::cout << "Compute B_angular tensor" << std::endl;

    TensorXs B(num_vertices * 3, 3, 3);
    B.setZero();

    constexpr auto kronecker_delta = [](int i, int j) -> ScalarType { return (i == j) ? 1.0 : 0.0; };
    const auto b_star_ijk = [&](const auto &r_hat, auto r, const auto &n, int i, int j, int k) {
        return -rho * (1. / (16. * M_PI * (1. - nu) * mu)) *
               (kronecker_delta(i, j) * (3. - 4. * nu) * r * n(k) - r_hat(i) * r_hat(k) * r * n(j));
    };

    const auto b_star = [&](const auto &y, const auto &x, const auto &n) {
        const auto r_vec = y - x;
        const auto r_hat = r_vec.stableNormalized().eval();
        const auto r = r_vec.norm();
        Eigen::Matrix<ScalarType, 3, 9> mat;
        for (int k = 0; k < 3; k++) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j <= i; j++) {
                    mat(i, 3 * k + j) = b_star_ijk(r_hat, r, n, i, j, k);
                    if (i != j) mat(j, 3 * k + i) = mat(i, 3 * k + j);
                }
            }
        }
        return mat;
    };

    if (use_galerkin) {
#pragma omp parallel for
        for (int f1 = 0; f1 < F.rows(); f1++) {
            for (int f2 = 0; f2 < F.rows(); f2++) {
                const auto n = N.row(f2);
                const auto integrand = [&](const auto &x, const auto &y, const auto &Lx, const auto &Ly) {
                    auto _b_star = b_star(y, x, n);
                    Eigen::Matrix<ScalarType, 9, 9, Eigen::RowMajor> res;
                    for (int i = 0; i < 3; i++) res.middleRows<3>(3 * i) = Lx(i) * _b_star;
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
                for (int k = 0; k < 3; k++) {
                    for (int i = 0; i < 3; i++) {
                        for (int p = 0; p < 3; p++) {
                            for (int q = 0; q < 3; q++) { B(3 * j1[i] + p, q, k) += mat(3 * i + p, 3 * k + q); }
                        }
                    }
                }
            }
        }
    } else {
#pragma omp parallel for
        for (Eigen::Index i = 0; i < num_vertices; i++) {
            const auto x = V.row(i);
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
                    const auto n = N.row(f_index);
                    const auto y1 = V.row(j1), y2 = V.row(j2), y3 = V.row(j3);
                    Eigen::Matrix<ScalarType, 3, 9> mat = integrate_collocation_nonsingular(
                        [&](const auto &y) { return b_star(y, x, n); }, y1, y2, y3, gaussian_quadrature_order,
                        quadrature_subdivision
                    );
                    for (int k = 0; k < 3; k++) {
                        for (int p = 0; p < 3; p++) {
                            for (int q = 0; q < 3; q++) { B(3 * i + p, q, k) += mat(p, 3 * k + q); }
                        }
                    }

                } else {
                    const auto n = N.row(f_index);
                    const Eigen::Index j1 = F(f_index, singular_vert_index),
                                       j2 = F(f_index, (singular_vert_index + 1) % 3),
                                       j3 = F(f_index, (singular_vert_index + 2) % 3);
                    const auto y1 = V.row(j1), y2 = V.row(j2), y3 = V.row(j3);

                    integrate_collocation_weakly_singular<ScalarType>(
                        [&](auto func) { // update loop
                            for (int p = 0; p < 3; p++) {
                                for (int q = 0; q <= p; q++) {
                                    for (int k = 0; k < 3; k++) {
                                        ScalarType elem = func(Eigen::Vector3i(p, q, k));
                                        B(3 * i + p, q, k) += elem;
                                        if (p != q) B(3 * i + p, q, k) += elem;
                                    }
                                }
                            }
                        },
                        [&](const auto &y, const auto &r_hat, const auto &r,
                            const auto &indices) -> ScalarType { // integrand
                            return r * b_star_ijk(r_hat, r, n, indices[0], indices[1], indices[2]);
                        },
                        y1, y2, y3, gaussian_quadrature_order, quadrature_subdivision
                    );
                }
            }
        }
    }

    for (Eigen::Index v = 0; v < num_vertices; v++) {
        auto x = V.row(v);
        Vector3s vec = x - cm;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) { B(3 * v + i, j, k) += B_trans(3 * v + i, j) * vec(k); }
            }
        }
    }

    return B;
}

MatrixX3s ElastodynamicsDRBEMMatrixCollection::compute_B_euler_matrix(const MatrixX3s &N, const TensorXs &B_angular) {
    std::cout << "Compute B_euler matrix" << std::endl;
    MatrixX3s B = MatrixX3s::Zero(num_vertices * 3, 3);

    TensorXs _B_angular;
    if (B_angular.size())
        _B_angular = B_angular;
    else
        load_tensor(_B_angular, coeffs_folder_path + "/B_angular.ten");

#pragma omp parallel for
    for (int i = 0; i < B.rows(); i++) {
        for (int j = 0; j < 3; j++) {
            B(i, j) = _B_angular(i, (j + 2) % 3, (j + 1) % 3) - _B_angular(i, (j + 1) % 3, (j + 2) % 3);
        }
    }

    return B;
}

VectorXs ElastodynamicsDRBEMMatrixCollection::compute_rotation_removal_weights(const MatrixX3s &N) {
    VectorXs res = VectorXs::Zero(num_vertices);

    VectorXs A;
    igl::doublearea(V, F, A);

    for (Eigen::Index f_index = 0; f_index < F.rows(); f_index++) {
        const Eigen::Index j1 = F(f_index, 0), j2 = F(f_index, 1), j3 = F(f_index, 2);
        const auto y1 = V.row(j1);
        const auto n = N.row(f_index);
        ScalarType val = (1. / 6.) * A(f_index) * y1.dot(n);
        res(j1) += val;
        res(j2) += val;
        res(j3) += val;
    }

    return res;
}