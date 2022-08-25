#ifndef __ELASTICS_BEM_MATRIX_COLLECTION_HPP__
#define __ELASTICS_BEM_MATRIX_COLLECTION_HPP__

#include "physics_base_matrix_collection.hpp"

class ElasticsBEMMatrixCollection : public PhysicsBaseMatrixCollection {
  protected:
    ElasticsBEMMatrixCollection();

  public:
    virtual ~ElasticsBEMMatrixCollection(){};

    virtual bool init(const nlohmann::json &config);

  protected:
    MatrixXs compute_elastostatic_H_matrix_collocation(const MatrixX3s &init_V, const MatrixX3s &N);
    MatrixXs compute_elastostatic_G_matrix_collocation(const MatrixX3s &init_V);
    MatrixXs compute_elastostatic_H_matrix_galerkin(const MatrixX3s &init_V, const MatrixX3s &N);
    MatrixXs compute_elastostatic_G_matrix_galerkin(const MatrixX3s &init_V);

    MatrixX3s compute_C_matrix_collocation(const MatrixX3s &N);
    SparseMatrix compute_C_matrix_galerkin(const MatrixX3s &N);

    ScalarType nu, mu;
    bool use_galerkin;
    std::string coeffs_folder_path;
    bool update_local_frame;
    ScalarType compression_ratio;

    size_t gaussian_quadrature_order;
    size_t quadrature_subdivision;
};

#endif //!__ELASTICS_BEM_MATRIX_COLLECTION_HPP__