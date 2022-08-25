#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#define __device__
#define __host__
#endif
#include <filesystem>
#include <thread>

#include "elastodynamics_cqbem_matrix_collection.hpp"
#include "matrix_io.hpp"

#include "elastodynamics_cqbem_collocation_kernels.cuh"
#include "elastodynamics_cqbem_galerkin_kernels.cuh"

#include "cuda_check_error.hpp"

void ElastodynamicsCQBEMMatrixCollection::compute_H_matrices(const MatrixX3s &V, const MatrixX3s &N, bool enable_cuda) {
    const double R = std::pow(1e-10, 1. / (2. * max_time_history));
    const size_t L = max_time_history + 1;
    const std::complex<double> exp_i2pi_L = std::exp(std::complex<double>(0.0, 1.0) * (2. * M_PI) / (double)L);
    const auto gamma_func =
        multistep_method == BDF1 ? [](const std::complex<double> &s) -> std::complex<double> { return 1. - s; }
    : [](const std::complex<double> &s) -> std::complex<double> { return 1.5 - 2. * s + s * s / 2.; };

#ifdef __CUDACC__
    cudaStream_t stream;
    ScalarType *dev_V, *dev_N;
    IntType *dev_F, *dev_vertex_map_inverse;
    ComplexType *dev_P;
    if (enable_cuda) {
        cuda_check_error([&]() { return cudaStreamCreate(&stream); });

        cuda_check_error([&]() { return cudaMallocAsync((void **)&dev_V, sizeof(ScalarType) * V.size(), stream); });
        cuda_check_error([&]() { return cudaMallocAsync((void **)&dev_F, sizeof(IntType) * F.size(), stream); });
        cuda_check_error([&]() { return cudaMallocAsync((void **)&dev_N, sizeof(ScalarType) * N.size(), stream); });
        cuda_check_error([&]() {
            return cudaMallocAsync((void **)&dev_P, sizeof(ComplexType) * num_vertices * 3 * num_vertices * 3, stream);
        });

        cuda_check_error([&]() {
            return cudaMemcpyAsync(dev_V, V.data(), sizeof(ScalarType) * V.size(), cudaMemcpyHostToDevice, stream);
        });
        cuda_check_error([&]() {
            return cudaMemcpyAsync(dev_F, F.data(), sizeof(IntType) * F.size(), cudaMemcpyHostToDevice, stream);
        });
        cuda_check_error([&]() {
            return cudaMemcpyAsync(dev_N, N.data(), sizeof(ScalarType) * N.size(), cudaMemcpyHostToDevice, stream);
        });

        if (enable_traction_discontinuity) {
            cuda_check_error([&]() {
                return cudaMallocAsync(
                    (void **)&dev_vertex_map_inverse, sizeof(IntType) * vertex_map_inverse.size(), stream
                );
            });
            cuda_check_error([&]() {
                return cudaMemcpyAsync(
                    dev_vertex_map_inverse, vertex_map_inverse.data(), sizeof(IntType) * vertex_map_inverse.size(),
                    cudaMemcpyHostToDevice, stream
                );
            });
        }
    }
#endif

    MatrixXc P(num_vertices * 3, num_vertices * 3);
    for (size_t l = 0; l < L / 2 + 1; l++) {
        std::cout << "Compute H hat matrices: " << l + 1 << "/" << L / 2 + 1 << std::endl;
        const std::complex<double> s = gamma_func(R * std::pow(exp_i2pi_L, l)) / (double)dt;

        if (use_galerkin) {
            if (enable_cuda) {
#ifdef __CUDACC__
                MatrixXc _P;
                std::thread cpu_thread([&]() {
                    _P = MatrixXc::Zero(num_vertices * 3, num_vertices * 3);
                    compute_elastodynamic_H_kernel_galerkin_singular(
                        _P.data(), V.data(), F.data(), N.data(), num_vertices, F.rows(), c1, c2,
                        std::complex<SingularScalarType>(s), gaussian_quadrature_order, quadrature_subdivision,
                        enable_traction_discontinuity ? vertex_map_inverse.data() : nullptr
                    );
                });

                cuda_check_error([&]() {
                    return cudaMemsetAsync(dev_P, 0, sizeof(ComplexType) * num_vertices * 3 * num_vertices * 3, stream);
                });
                size_t _grid = (F.rows() + cuda_thread_per_block - 1) / cuda_thread_per_block;
                size_t _block = cuda_thread_per_block;
                dim3 grid(_grid, _grid);
                dim3 block(_block, _block);
                compute_elastodynamic_H_kernel_galerkin_non_singular_global_wrapper<<<grid, block, 0, stream>>>(
                    dev_P, dev_V, dev_F, dev_N, num_vertices, F.rows(), c1, c2, s, gaussian_quadrature_order,
                    quadrature_subdivision, enable_traction_discontinuity ? vertex_map_inverse.data() : nullptr
                );
                cuda_check_last_error();
                cuda_check_error([&]() {
                    return cudaMemcpyAsync(
                        P.data(), dev_P, sizeof(ComplexType) * P.size(), cudaMemcpyDeviceToHost, stream
                    );
                });
                cuda_check_error([&]() { return cudaStreamSynchronize(stream); });

                cpu_thread.join();
                P += _P;
#endif
            } else {
                P.setZero();
#pragma omp parallel for
                for (Eigen::Index f1 = 0; f1 < F.rows(); f1++) {
                    for (Eigen::Index f2 = 0; f2 < F.rows(); f2++) {
                        compute_elastodynamic_H_kernel_galerkin_non_singular(
                            P.data(), V.data(), F.data(), N.data(), num_vertices, F.rows(), c1, c2,
                            complex<NonSingularScalarType>(s), gaussian_quadrature_order, quadrature_subdivision, f1,
                            f2, enable_traction_discontinuity ? vertex_map_inverse.data() : nullptr
                        );
                    }
                }
                compute_elastodynamic_H_kernel_galerkin_singular(
                    P.data(), V.data(), F.data(), N.data(), num_vertices, F.rows(), c1, c2,
                    std::complex<SingularScalarType>(s), gaussian_quadrature_order, quadrature_subdivision,
                    enable_traction_discontinuity ? vertex_map_inverse.data() : nullptr
                );
            }
        } else {
            if (enable_cuda) {
#ifdef __CUDACC__
                size_t grid = (num_vertices + cuda_thread_per_block - 1) / cuda_thread_per_block;
                size_t block = cuda_thread_per_block;
                compute_elastodynamic_H_kernel_collocation_global_wrapper<<<grid, block, 0, stream>>>(
                    dev_P, dev_V, dev_F, dev_N, num_vertices, F.rows(), c1, c2, s, gaussian_quadrature_order,
                    quadrature_subdivision, cuda_thread_per_block,
                    enable_traction_discontinuity ? dev_vertex_map_inverse : nullptr
                );
                cuda_check_last_error();
                cuda_check_error([&]() {
                    return cudaMemcpyAsync(
                        P.data(), dev_P, sizeof(ComplexType) * P.size(), cudaMemcpyDeviceToHost, stream
                    );
                });
                cuda_check_error([&]() { return cudaStreamSynchronize(stream); });
#endif
            } else {
#pragma omp parallel for
                for (Eigen::Index i = 0; i < num_vertices; i++) {
                    compute_elastodynamic_H_kernel_collocation(
                        P.data(), V.data(), F.data(), N.data(), num_vertices, F.rows(), c1, c2, s,
                        gaussian_quadrature_order, quadrature_subdivision, i,
                        enable_traction_discontinuity ? vertex_map_inverse.data() : nullptr
                    );
                }
            }
        }

        Eigen::save_matrix(P, coeffs_folder_path + "/P_" + std::to_string(l) + ".mat");
    }

#ifdef __CUDACC__
    if (enable_cuda) {
        cuda_check_error([&]() { return cudaFreeAsync(dev_V, stream); });
        cuda_check_error([&]() { return cudaFreeAsync(dev_F, stream); });
        cuda_check_error([&]() { return cudaFreeAsync(dev_N, stream); });
        cuda_check_error([&]() { return cudaFreeAsync(dev_P, stream); });
        if (enable_traction_discontinuity)
            cuda_check_error([&]() { return cudaFreeAsync(dev_vertex_map_inverse, stream); });

        cuda_check_error([&]() { return cudaStreamSynchronize(stream); });
        cuda_check_error([&]() { return cudaStreamDestroy(stream); });
    }
#endif

    for (size_t n = 0; n < L; n++) {
        std::cout << "Compute H matrices: " << n + 1 << "/" << L << std::endl;
        MatrixXs _H = MatrixXs::Zero(num_vertices * 3, num_vertices * 3);
        for (size_t l = 0; l < L / 2 + 1; l++) {
            MatrixXc P;
            Eigen::load_matrix(P, coeffs_folder_path + "/P_" + std::to_string(l) + ".mat");
            _H += (P * ComplexType(std::pow(exp_i2pi_L, -(double)n * l))).real();
            if (l != 0 && (l != L / 2 || L % 2 == 1))
                _H += (P.conjugate() * ComplexType(std::pow(exp_i2pi_L, -(double)n * (L - l)))).real();
        }
        _H *= (ScalarType)(std::pow(R, -(double)n) / (double)L);

        if (n == 0) {
            if (use_galerkin) {
                _H += compute_C_matrix_galerkin(N);
            } else {
                MatrixX3s C = compute_C_matrix_collocation(N);
                for (Eigen::Index i = 0; i < num_vertices; i++) {
                    _H.block<3, 3>(3 * i, 3 * i) += C.middleRows<3>(3 * i);
                }
            }
        }

        Eigen::save_matrix(_H, coeffs_folder_path + "/H_" + std::to_string(n) + ".mat");
    }

    for (size_t l = 0; l < L / 2 + 1; l++)
        std::filesystem::remove(coeffs_folder_path + "/P_" + std::to_string(l) + ".mat");
}
