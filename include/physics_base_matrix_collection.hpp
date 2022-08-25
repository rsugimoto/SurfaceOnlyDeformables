#ifndef __PHYSICS_BASE_MATRIX_COLLECTION__
#define __PHYSICS_BASE_MATRIX_COLLECTION__

#include "json.hpp"
#include "type_declaration.hpp"

#include "wavelets.hpp"

class ConstraintsSolver;

class PhysicsBaseMatrixCollection {
  protected:
    PhysicsBaseMatrixCollection();

  public:
    virtual ~PhysicsBaseMatrixCollection(){};
    virtual bool init(const nlohmann::json &config);

  protected:
    bool enable_traction_discontinuity;

    const MatrixX3s &get_V() const { return V; };
    const MatrixX3i &get_F() const { return F; };

    Matrix3s compute_inertia_tensor(const MatrixX3s &N);
    Matrix3Xs compute_external_translational_acc_matrix(const ScalarType mass);
    Matrix3Xs compute_external_rotational_acc_matrix(const Matrix3s &I);

    MatrixX3s V;
    MatrixX3i F;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm;
    std::vector<std::vector<Eigen::Index>> VF;
    Eigen::Index num_vertices, original_num_vertices;

    // These three are used only when traction discontinuity is enabled.
    VectorXi vertex_map_inverse; // vertex_map_inverse[new_idx] = old_idx
    MatrixX3s original_V;
    MatrixX3i original_F;

    ScalarType rho, mass;
    RowVector3s cm, original_cm;
    Matrix3s I;
    Matrix3Xs translational_acc_mat, rotational_acc_mat;

    CompressedMatrix<> u_p_compressed;
    MatrixXs u_p;
    SparseMatrix u_p_diag;
    PermutationMatrix compression_permutation;

    bool use_compressed_matrices;

  private:
    bool load_mesh(const nlohmann::json &config);

    friend class PhysicsBaseObject;
    friend class ConstraintsSolver;
};

#endif //!__PHYSICS_BASE_MATRIX_COLLECTION__