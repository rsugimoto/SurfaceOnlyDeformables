#ifndef __ELASTODYNAMICS_DRBEM_MATRIX_COLLECTION_HPP__
#define __ELASTODYNAMICS_DRBEM_MATRIX_COLLECTION_HPP__

#include "elastics_bem_matrix_collection.hpp"
#include "wavelets.hpp"

class ElastodynamicsDRBEMMatrixCollection : public ElasticsBEMMatrixCollection {
  public:
    ElastodynamicsDRBEMMatrixCollection(ScalarType dt);
    ~ElastodynamicsDRBEMMatrixCollection();

    bool init(const nlohmann::json &config) override;

  private:
    MatrixXs compute_M_matrix(const MatrixXs &P, const MatrixXs &U);
    MatrixX3s compute_B_trans_matrix(const MatrixX3s &N);
    TensorXs compute_B_angular_tensor(const MatrixX3s &N, const RowVector3s &cm, const MatrixX3s &B_trans);
    MatrixX3s compute_B_euler_matrix(const MatrixX3s &N, const TensorXs &B_angular);

    VectorXs compute_rotation_removal_weights(const MatrixX3s &N);

    ScalarType dt;

  protected:
    CompressedMatrix<> M_compressed;
    MatrixXs M;
    MatrixXs B_trans_inv_H, B_trans_inv_G, H_inv_B_trans, B_euler_inv_H, B_euler_inv_G, H_inv_B_euler,
        B_euler_inv_B_trans;
    Eigen::CompleteOrthogonalDecomposition<MatrixXs> B_trans_inv, B_euler_inv;

    MatrixX3s B_trans, B_euler;
    CompressedMatrix<> u_b_compressed;
    MatrixXs u_b;
    VectorXs rotation_removal_weights;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    friend class ElastodynamicsDRBEMObject;
};

#endif //!__ELASTODYNAMICS_DRBEM_MATRIX_COLLECTION_HPP__