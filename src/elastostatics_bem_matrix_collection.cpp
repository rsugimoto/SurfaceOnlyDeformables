#include "elastostatics_bem_matrix_collection.hpp"

#include <filesystem>
#include <iostream>
#include <sstream>

#include <igl/per_face_normals.h>

#include "cross_product_matrix.hpp"
#include "matrix_io.hpp"

ElastostaticsBEMMatrixCollection::ElastostaticsBEMMatrixCollection() : ElasticsBEMMatrixCollection() {}
ElastostaticsBEMMatrixCollection::~ElastostaticsBEMMatrixCollection() {}

bool ElastostaticsBEMMatrixCollection::init(const nlohmann::json &config) {
    if (!ElasticsBEMMatrixCollection::init(config)) return false;

    std::cout << "type: Elastostatics BEM" << std::endl;

    if (nu >= 0.5) {
        std::cout << "nu>=0.5 is not supported with elastostatics BEM" << std::endl;
        return false;
    }

    // Use hashing to store matrices for different configs in separate folders.
    coeffs_folder_path = config["simulation"]["coeffs_folder_path"].get<std::string>() + "/elastostaticsBEM_";
    {
        nlohmann::json config_copy = config;
        const char *configs_ignored[] = {"compression_ratio", "update_local_frame"};
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

    // load u_p or compute u_p
    if (!Eigen::load_matrix(u_p, coeffs_folder_path + "/u_p.mat")) {
        MatrixX3s N;
        igl::per_face_normals(V, F, N);

        MatrixXs _H, _G;
        if (use_galerkin) {
            _H = compute_elastostatic_H_matrix_galerkin(V, N);
            _G = compute_elastostatic_G_matrix_galerkin(V);
        } else {
            _H = compute_elastostatic_H_matrix_collocation(V, N);
            _G = compute_elastostatic_G_matrix_collocation(V);
        }

        // regularization based on [Hahn and Wojtan 2016], but is applied on H instead of K.
        Eigen::Matrix<ScalarType, 6, Eigen::Dynamic> T =
            Eigen::Matrix<ScalarType, 6, Eigen::Dynamic>::Zero(6, num_vertices * 3);
        {
            VectorXs A;
            igl::doublearea(V, F, A);

            auto T_d = T.topRows<3>();
            for (Eigen::Index e = 0; e < F.rows(); e++) {
                for (int i = 0; i < 3; i++) { T_d.middleCols<3>(3 * F(e, i)).diagonal().array() += A(e) / 6.; }
            }

            auto T_r = T.bottomRows<3>();
            for (Eigen::Index e = 0; e < F.rows(); e++) {
                for (int i = 0; i < 3; i++) {
                    T_r.middleCols<3>(3 * F(e, i)) +=
                        A(e) / 6. * 0.25 *
                        cross_product_matrix(V.row(F(e, 0)) + V.row(F(e, 1)) + V.row(F(e, 2)) + V.row(F(e, i)));
                }
            }
        }
        MatrixXs H = _H.transpose() * _H + std::pow(_H.trace() / (3 * num_vertices), 2.) * (T.transpose() * T);

        // precompute the G matrix s.t. the traction becomes global force- or traction-free.
        auto &A = T; // because we use piece-wise linear interp. for p, A and T are the same.
        MatrixXs G = _H.transpose() * _G *
                     (MatrixXs::Identity(num_vertices * 3, num_vertices * 3) -
                      A.transpose() * (A * A.transpose()).partialPivLu().solve(A));
        _H.resize(0, 0); // free memory

        // H is symmetric
        u_p = H.ldlt().solve(G);
        Eigen::save_matrix(u_p, coeffs_folder_path + "/u_p.mat");
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
        if (!load_compressed_matrix(
                u_p_compressed, coeffs_folder_path + "/u_p_" + std::to_string(compression_ratio) + ".cpmat"
            )) {
            std::cout << "Compressing u_p matrix..." << std::endl;
            u_p_compressed = CompressedMatrix<>(
                compression_permutation * u_p * compression_permutation.transpose(), compression_ratio
            );
            save_compressed_matrix(
                u_p_compressed, coeffs_folder_path + "/u_p_" + std::to_string(compression_ratio) + ".cpmat"
            );
            std::cout << "Finished compressing u_p matrix." << std::endl;
        }

        u_p.resize(0, 0); // free memory
    }

    return true;
}
