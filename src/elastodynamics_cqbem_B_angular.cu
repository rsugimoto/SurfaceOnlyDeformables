#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#define __device__
#define __host__
#endif
#include <thread>

#include "elastodynamics_cqbem_matrix_collection.hpp"

#include "elastodynamics_cqbem_collocation_kernels.cuh"
#include "elastodynamics_cqbem_galerkin_kernels.cuh"

void ElastodynamicsCQBEMMatrixCollection::compute_B_angular_tensors(
    const MatrixX3s &V, const MatrixX3s &N, const RowVector3s &cm, const std::vector<MatrixX3s> &B_trans,
    std::vector<TensorXs> &B_angular, bool enable_cuda
) {
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
    ComplexType *dev_B;
    if (enable_cuda) {
        cudaStreamCreate(&stream);

        cudaMallocAsync((void **)&dev_V, sizeof(ScalarType) * V.size(), stream);
        cudaMallocAsync((void **)&dev_F, sizeof(IntType) * F.size(), stream);
        cudaMallocAsync((void **)&dev_N, sizeof(ScalarType) * N.size(), stream);
        cudaMallocAsync((void **)&dev_B, sizeof(ComplexType) * num_vertices * 3 * 3 * 3, stream);

        cudaMemcpyAsync(dev_V, V.data(), sizeof(ScalarType) * V.size(), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dev_F, F.data(), sizeof(IntType) * F.size(), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dev_N, N.data(), sizeof(ScalarType) * N.size(), cudaMemcpyHostToDevice, stream);

        if (enable_traction_discontinuity) {
            cudaMallocAsync((void **)&dev_vertex_map_inverse, sizeof(IntType) * vertex_map_inverse.size(), stream);
            cudaMemcpyAsync(
                dev_vertex_map_inverse, vertex_map_inverse.data(), sizeof(IntType) * vertex_map_inverse.size(),
                cudaMemcpyHostToDevice, stream
            );
        }
    }
#endif

    TensorXc B(num_vertices * 3, 3, 3);
    std::vector<TensorXc> B_list;
    for (size_t l = 0; l < L / 2 + 1; l++) {
        std::cout << "Compute B_angular hat tensors: " << l + 1 << "/" << L / 2 + 1 << std::endl;
        const ComplexType s = ComplexType(gamma_func(R * std::pow(exp_i2pi_L, l)) / (double)dt);

        if (use_galerkin) {
            if (enable_cuda) {
#ifdef __CUDACC__
                TensorXc _B(num_vertices * 3, 3, 3);
                std::thread cpu_thread([&]() {
                    _B.setZero();
                    compute_elastodynamic_B_angular_kernel_galerkin_singular(
                        _B.data(), V.data(), F.data(), N.data(), num_vertices, F.rows(), c1, c2,
                        std::complex<SingularScalarType>(s), gaussian_quadrature_order, quadrature_subdivision,
                        enable_traction_discontinuity ? vertex_map_inverse.data() : nullptr
                    );
                });

                cudaMemsetAsync(dev_B, 0, sizeof(ComplexType) * num_vertices * 3 * 3 * 3, stream);
                size_t _grid = (F.rows() + cuda_thread_per_block - 1) / cuda_thread_per_block;
                size_t _block = cuda_thread_per_block;
                dim3 grid(_grid, _grid);
                dim3 block(_block, _block);
                compute_elastodynamic_B_angular_kernel_galerkin_non_singular_global_wrapper<<<grid, block>>>(
                    dev_B, dev_V, dev_F, dev_N, num_vertices, F.rows(), c1, c2, s, gaussian_quadrature_order,
                    quadrature_subdivision, enable_traction_discontinuity ? dev_vertex_map_inverse : nullptr
                );
                cudaError_t error = cudaGetLastError();
                if (error != cudaSuccess) std::cout << cudaGetErrorString(error) << std::endl;
                cudaMemcpyAsync(B.data(), dev_B, sizeof(ComplexType) * B.size(), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);

                cpu_thread.join();
                B += _B;
#endif
            } else {
                B.setZero();

#pragma omp parallel for
                for (Eigen::Index f1 = 0; f1 < F.rows(); f1++) {
                    for (Eigen::Index f2 = 0; f2 < F.rows(); f2++) {
                        compute_elastodynamic_B_angular_kernel_galerkin_non_singular(
                            B.data(), V.data(), F.data(), N.data(), num_vertices, F.rows(), c1, c2,
                            complex<NonSingularScalarType>(s), gaussian_quadrature_order, quadrature_subdivision, f1,
                            f2, enable_traction_discontinuity ? vertex_map_inverse.data() : nullptr
                        );
                    }
                }

                compute_elastodynamic_B_angular_kernel_galerkin_singular(
                    B.data(), V.data(), F.data(), N.data(), num_vertices, F.rows(), c1, c2,
                    std::complex<SingularScalarType>(s), gaussian_quadrature_order, quadrature_subdivision,
                    enable_traction_discontinuity ? vertex_map_inverse.data() : nullptr
                );
            }
        } else {
            if (enable_cuda) {
#ifdef __CUDACC__
                size_t grid = (num_vertices + cuda_thread_per_block - 1) / cuda_thread_per_block;
                size_t block = cuda_thread_per_block;
                compute_elastodynamic_B_angular_kernel_collocation_global_wrapper<<<grid, block>>>(
                    dev_B, dev_V, dev_F, dev_N, num_vertices, F.rows(), c1, c2, s, gaussian_quadrature_order,
                    quadrature_subdivision, cuda_thread_per_block,
                    enable_traction_discontinuity ? dev_vertex_map_inverse : nullptr
                );
                cudaError_t error = cudaGetLastError();
                if (error != cudaSuccess) std::cout << cudaGetErrorString(error) << std::endl;
                cudaMemcpyAsync(B.data(), dev_B, sizeof(ComplexType) * B.size(), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
#endif
            } else {
#pragma omp parallel for
                for (Eigen::Index i = 0; i < num_vertices; i++) {
                    compute_elastodynamic_B_angular_kernel_collocation(
                        B.data(), V.data(), F.data(), N.data(), num_vertices, F.rows(), c1, c2, s,
                        gaussian_quadrature_order, quadrature_subdivision, i,
                        enable_traction_discontinuity ? vertex_map_inverse.data() : nullptr
                    );
                }
            }
        }

        B_list.push_back(B);
    }

#ifdef __CUDACC__
    if (enable_cuda) {
        cudaFreeAsync(dev_V, stream);
        cudaFreeAsync(dev_F, stream);
        cudaFreeAsync(dev_N, stream);
        cudaFreeAsync(dev_B, stream);
        if (enable_traction_discontinuity) cudaFreeAsync(dev_vertex_map_inverse, stream);

        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
#endif

    for (size_t n = 0; n < L; n++) {
        std::cout << "Compute B_angular tensors: " << n + 1 << "/" << L << std::endl;
        TensorXs B = TensorXs(num_vertices * 3, 3, 3);
        B.setZero();

#pragma omp parallel for
        for (size_t l = 0; l < L / 2 + 1; l++) {
            const TensorXc &B_hat = B_list[l];
            TensorXs B_hat_real = (B_hat * ComplexType(std::pow(exp_i2pi_L, -(double)n * l))).real();
            if (l != 0 && (l != L / 2 || L % 2 == 1))
                B_hat_real += (B_hat.conjugate() * ComplexType(std::pow(exp_i2pi_L, -(double)n * (L - l)))).real();
#pragma omp critical
            { B += B_hat_real; }
        }
        B = B * (ScalarType)((std::pow(R, -(double)n) / (double)L));

        for (Eigen::Index v = 0; v < num_vertices; v++) {
            auto x = V.row(v);
            Vector3s vec = x - cm;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) { B(3 * v + i, j, k) += B_trans[n](3 * v + i, j) * vec(k); }
                }
            }
        }

        B_angular.push_back(B);
    }
}