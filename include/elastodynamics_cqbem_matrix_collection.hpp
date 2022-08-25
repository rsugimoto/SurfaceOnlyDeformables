#ifndef __ELASTODYNAMICS_CQBEM_MATRIX_COLLECTION_HPP__
#define __ELASTODYNAMICS_CQBEM_MATRIX_COLLECTION_HPP__

#include "elastics_bem_matrix_collection.hpp"
#include "wavelets.hpp"
#include <vector>

enum MultistepMethod { BDF1, BDF2 };

class ElastodynamicsCQBEMMatrixCollection : public ElasticsBEMMatrixCollection {
  public:
    ElastodynamicsCQBEMMatrixCollection(ScalarType dt, bool enable_cuda, unsigned int cuda_thread_per_block);
    ~ElastodynamicsCQBEMMatrixCollection();

    bool init(const nlohmann::json &config) override;

  private:
    void compute_G_matrices(const MatrixX3s &init_V, bool enable_cuda);
    void compute_H_matrices(const MatrixX3s &init_V, const MatrixX3s &N, bool enable_cuda);
    void compute_B_trans_matrices(
        const MatrixX3s &init_V, const MatrixX3s &N, std::vector<MatrixX3s> &B_trans, bool enable_cuda
    );
    void compute_B_angular_tensors(
        const MatrixX3s &init_V, const MatrixX3s &N, const RowVector3s &cm, const std::vector<MatrixX3s> &B_trans,
        std::vector<TensorXs> &B_angular, bool enable_cuda
    );
    void compute_B_euler_matrices(
        const MatrixX3s &N, const std::vector<TensorXs> &B_angular, std::vector<MatrixX3s> &B_euler
    );

    VectorXs compute_rotation_removal_weights(const MatrixX3s &N);

    bool enable_cuda;
    unsigned int cuda_thread_per_block;
    MultistepMethod multistep_method;
    ScalarType c1, c2, dt;

  protected:
    CompressedMatrix<> H_inv_compressed;
    Eigen::PartialPivLU<MatrixXs> H_inv;
    MatrixXs B_trans_inv_H, B_trans_inv_G_prime, H_inv_B_trans, B_euler_inv_H, B_euler_inv_G_prime, H_inv_B_euler,
        B_euler_inv_B_trans;
    Eigen::CompleteOrthogonalDecomposition<MatrixXs> B_trans_inv, B_euler_inv;

    std::vector<MatrixX6s> B_matrices;
    std::vector<CompressedMatrix<>> G_compressed_matrices, H_compressed_matrices;
    std::vector<MatrixXs> G_matrices, H_matrices;

    size_t max_time_history;

    VectorXs rotation_removal_weights;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    friend class ElastodynamicsCQBEMObject;
};

#endif //!__ELASTODYNAMICS_CQBEM_MATRIX_COLLECTION_HPP__