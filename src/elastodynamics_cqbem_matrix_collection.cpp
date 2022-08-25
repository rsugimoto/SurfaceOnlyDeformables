#include "elastodynamics_cqbem_matrix_collection.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <sstream>

#include <igl/edge_lengths.h>
#include <igl/per_face_normals.h>

#include "matrix_io.hpp"

ElastodynamicsCQBEMMatrixCollection::ElastodynamicsCQBEMMatrixCollection(
    ScalarType dt, bool enable_cuda, unsigned int cuda_thread_per_block
)
    : ElasticsBEMMatrixCollection(), dt(dt), enable_cuda(enable_cuda), cuda_thread_per_block(cuda_thread_per_block) {}

ElastodynamicsCQBEMMatrixCollection::~ElastodynamicsCQBEMMatrixCollection() {}

bool ElastodynamicsCQBEMMatrixCollection::init(const nlohmann::json &config) {
    if (!ElasticsBEMMatrixCollection::init(config)) return false;

    std::cout << "type: Elastodynamics CQBEM" << std::endl;

    if (dt != config["simulation"]["dt"].get<ScalarType>()) {
        std::cout << "dt for simulator and elastodynamics BEM need to be the same." << std::endl;
        return false;
    }

    std::string multistep_method_str = config["simulation"]["multistep_method"].get<std::string>();
    if (multistep_method_str == "BDF1")
        multistep_method = BDF1;
    else if (multistep_method_str == "BDF2")
        multistep_method = BDF2;
    else {
        std::cout << "Unsupported multistep method detected. Valid options are \"BDF1\"/\"BDF2\"" << std::endl;
        return false;
    }

    // Use hashing to store matrices for different configs in separate folders.
    coeffs_folder_path = config["simulation"]["coeffs_folder_path"].get<std::string>() + "/elastodynamicsCQBEM_";
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

    ScalarType lambda = 2 * this->nu * this->mu / (1. - 2. * this->nu);
    c1 = std::sqrt((lambda + 2. * this->mu) / this->rho);
    c2 = std::sqrt(this->mu / this->rho);

    ScalarType max_distance = 0.0;
    for (int j = 0; j < num_vertices; j++) {
        for (int i = 0; i < num_vertices; i++) {
            max_distance = std::max(max_distance, (V.row(i) - V.row(j)).stableNorm());
        }
    }
    max_time_history = std::floor(max_distance / (c2 * dt) + 2);

    std::cout << "c1: " << c1 << ", c2: " << c2 << ", max_time_history: " << max_time_history << std::endl;

    MatrixX3s N;
    igl::per_face_normals(V, F, N);

    // compute H and G matrices if they are not available in the filesystem.
    {
        bool compute_coeff_matrices = false;
        for (size_t k = 0; k < max_time_history + 1; k++) {
            if (!std::filesystem::exists(coeffs_folder_path + "/H_" + std::to_string(k) + ".mat")) {
                compute_coeff_matrices = true;
                break;
            }
            if (!std::filesystem::exists(coeffs_folder_path + "/G_" + std::to_string(k) + ".mat")) {
                compute_coeff_matrices = true;
                break;
            }
        }

        if (compute_coeff_matrices) {
            compute_H_matrices(V, N, enable_cuda);
            compute_G_matrices(V, enable_cuda);
        }
    }

    // load compressed/uncompressed H and G matrices
    for (size_t k = 0; k < max_time_history + 1; k++) {
        if (use_compressed_matrices) {
            {
                H_compressed_matrices.emplace_back();
                CompressedMatrix<> &H_compressed = H_compressed_matrices.back();
                if (!load_compressed_matrix(
                        H_compressed, coeffs_folder_path + "/H_" + std::to_string(k) + "_" +
                                          std::to_string(compression_ratio) + ".cpmat"
                    )) {
                    std::cout << "Compressing H_" << k + 1 << " matrix..." << std::endl;
                    MatrixXs H;
                    Eigen::load_matrix(H, coeffs_folder_path + "/H_" + std::to_string(k) + ".mat");
                    H_compressed = CompressedMatrix<>(
                        compression_permutation * -H * compression_permutation.transpose(), compression_ratio
                    );
                    save_compressed_matrix(
                        H_compressed, coeffs_folder_path + "/H_" + std::to_string(k) + "_" +
                                          std::to_string(compression_ratio) + ".cpmat"
                    );
                }
            }
            {
                G_compressed_matrices.emplace_back();
                CompressedMatrix<> &G_compressed = G_compressed_matrices.back();
                if (!load_compressed_matrix(
                        G_compressed, coeffs_folder_path + "/G_" + std::to_string(k) + "_" +
                                          std::to_string(compression_ratio) + ".cpmat"
                    )) {
                    std::cout << "Compressing G_" << k + 1 << " matrix..." << std::endl;
                    MatrixXs G;
                    Eigen::load_matrix(G, coeffs_folder_path + "/G_" + std::to_string(k) + ".mat");
                    G_compressed = CompressedMatrix<>(
                        compression_permutation * G * compression_permutation.transpose(), compression_ratio
                    );
                    save_compressed_matrix(
                        G_compressed, coeffs_folder_path + "/G_" + std::to_string(k) + "_" +
                                          std::to_string(compression_ratio) + ".cpmat"
                    );
                }
            }
        } else {
            H_matrices.emplace_back();
            Eigen::load_matrix(H_matrices.back(), coeffs_folder_path + "/H_" + std::to_string(k) + ".mat");
            H_matrices.back() *= -1.;
            G_matrices.emplace_back();
            Eigen::load_matrix(G_matrices.back(), coeffs_folder_path + "/G_" + std::to_string(k) + ".mat");
        }
    }

    MatrixX3s B_trans, B_euler;
    // load/compute B matrices
    {
        bool load_coeff_matrices = true;
        for (size_t k = 0; k < max_time_history + 1; k++)
            if (!std::filesystem::exists(coeffs_folder_path + "/B_" + std::to_string(k) + ".mat")) {
                load_coeff_matrices = false;
                break;
            }

        if (load_coeff_matrices) {
            for (size_t k = 0; k < max_time_history + 1; k++) {
                MatrixXs B;
                Eigen::load_matrix(B, coeffs_folder_path + "/B_" + std::to_string(k) + ".mat");
                B_matrices.push_back(B);
            }
        } else {
            std::vector<MatrixX3s> B_trans, B_euler;
            std::vector<TensorXs> B_angular;
            compute_B_trans_matrices(V, N, B_trans, enable_cuda);
            compute_B_angular_tensors(V, N, cm, B_trans, B_angular, enable_cuda);
            compute_B_euler_matrices(N, B_angular, B_euler);
            for (size_t k = 0; k < max_time_history + 1; k++) {
                B_matrices.emplace_back(B_trans[k].rows(), 6);
                B_matrices.back() << B_trans[k], B_euler[k];
                Eigen::save_matrix(B_matrices.back(), coeffs_folder_path + "/B_" + std::to_string(k) + ".mat");
            }
        }

        B_trans = B_matrices[0].leftCols<3>();
        B_euler = B_matrices[0].rightCols<3>();
    }

    MatrixXs H, G_prime;
    Eigen::load_matrix(H, coeffs_folder_path + "/H_0.mat");
    Eigen::load_matrix(G_prime, coeffs_folder_path + "/G_0.mat");
    if (update_local_frame) G_prime += B_trans * translational_acc_mat + B_euler * rotational_acc_mat;

    B_trans_inv.compute(B_trans);
    B_euler_inv.compute(B_euler);

    B_trans_inv_H = B_trans_inv.solve(H);
    B_euler_inv_H = B_euler_inv.solve(H);
    B_trans_inv_G_prime = B_trans_inv.solve(G_prime);
    B_euler_inv_G_prime = B_euler_inv.solve(G_prime);
    B_euler_inv_B_trans = B_euler_inv.solve(B_trans);
    rotation_removal_weights = compute_rotation_removal_weights(N);

    bool H_inv_initialized = false;
    if (!Eigen::load_matrix(
            u_p, coeffs_folder_path + "/u_p" + (update_local_frame ? "" : "no_frame_update_") + ".mat"
        ) ||
        !Eigen::load_matrix(H_inv_B_trans, coeffs_folder_path + "/H_inv_B_trans.mat") ||
        !Eigen::load_matrix(H_inv_B_euler, coeffs_folder_path + "/H_inv_B_euler.mat") ||
        !std::filesystem::exists(coeffs_folder_path + "/H_inv.mat")) {
        std::cout << "Computing H inverse..." << std::endl;
        H_inv.compute(H);
        H_inv_initialized = true;
        std::cout << "Finished computing H inverse" << std::endl;

        u_p = H_inv.solve(G_prime);
        H_inv_B_trans = H_inv.solve(B_trans);
        H_inv_B_euler = H_inv.solve(B_euler);
        MatrixXs H_inv_mat = H_inv.inverse();

        Eigen::save_matrix(u_p, coeffs_folder_path + "/u_p" + (update_local_frame ? "" : "no_frame_update_") + ".mat");
        Eigen::save_matrix(H_inv_B_trans, coeffs_folder_path + "/H_inv_B_trans.mat");
        Eigen::save_matrix(H_inv_B_euler, coeffs_folder_path + "/H_inv_B_euler.mat");
        Eigen::save_matrix(H_inv_mat, coeffs_folder_path + "/H_inv.mat");
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

        // H_inv
        {
            if (!load_compressed_matrix(
                    H_inv_compressed, coeffs_folder_path + "/H_inv_" + std::to_string(compression_ratio) + ".cpmat"
                )) {
                MatrixXs H_inv_mat = H_inv.inverse();
                H_inv.compute(MatrixXs(0, 0)); // free memory

                std::cout << "Compressing H inverse matrix..." << std::endl;
                H_inv_compressed = CompressedMatrix<>(
                    compression_permutation * H_inv_mat * compression_permutation.transpose(), compression_ratio
                );
                save_compressed_matrix(
                    H_inv_compressed, coeffs_folder_path + "/H_inv_" + std::to_string(compression_ratio) + ".cpmat"
                );
                std::cout << "Finished compressing H inverse matrix." << std::endl;

                H_inv_mat.resize(0, 0); // free memory
            }
        }

    } else if (!H_inv_initialized) {
        std::cout << "Computing H inverse matrix..." << std::endl;
        H_inv.compute(H);
        std::cout << "Finished computing H inverse" << std::endl;
    }

    return true;
}

void ElastodynamicsCQBEMMatrixCollection::compute_B_euler_matrices(
    const MatrixX3s &N, const std::vector<TensorXs> &B_angular, std::vector<MatrixX3s> &B_euler
) {
    const size_t L = max_time_history + 1;

    for (size_t n = 0; n < L; n++) {
        std::cout << "Compute B_euler matrices: " << n + 1 << "/" << L << std::endl;
        MatrixX3s B = MatrixX3s::Zero(num_vertices * 3, 3);

        TensorXs _B_angular;

        if (B_angular.size())
            _B_angular = B_angular[n];
        else
            load_tensor(_B_angular, coeffs_folder_path + "/B_angular_" + std::to_string(n) + ".ten");

#pragma omp parallel for
        for (int i = 0; i < B.rows(); i++) {
            for (int j = 0; j < 3; j++) {
                B(i, j) = _B_angular(i, (j + 2) % 3, (j + 1) % 3) - _B_angular(i, (j + 1) % 3, (j + 2) % 3);
            }
        }

        B_euler.push_back(B);
    }
}

VectorXs ElastodynamicsCQBEMMatrixCollection::compute_rotation_removal_weights(const MatrixX3s &N) {
    VectorXs res = VectorXs::Zero(num_vertices);

    VectorXs A;
    igl::doublearea(V, F, A);

    for (Eigen::Index f_index = 0; f_index < F.rows(); f_index++) {
        const Eigen::Index j1 = F(f_index, 0), j2 = F(f_index, 1), j3 = F(f_index, 2);
        const auto y1 = V.row(j1);
        const auto &n = N.row(f_index);
        ScalarType val = (1. / 6.) * A(f_index) * y1.dot(n);
        res(j1) += val;
        res(j2) += val;
        res(j3) += val;
    }
    return res;
}
