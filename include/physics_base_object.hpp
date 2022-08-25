#ifndef __PHYSICS_BASE_OBJECT_HPP__
#define __PHYSICS_BASE_OBJECT_HPP__

#include "physics_base_matrix_collection.hpp"
#include <memory>
#include <vector>

enum FrictionCombine { Average = 0, Minimum, Multiply, Maximum };

class PhysicsBaseMatrixCollection;

class PhysicsBaseObject {
  public:
    PhysicsBaseObject(
        const std::shared_ptr<const PhysicsBaseMatrixCollection> matrix_collection, ScalarType dt,
        const Vector3s &gravitational_constant
    );
    virtual ~PhysicsBaseObject(){};

    virtual bool init(const nlohmann::json &config);

    const MatrixX3i &get_F() const { return matrix_collection->F; };
    const MatrixX3s &get_V_local() const { return V_local; };
    const RowVector3s &get_cm() const { return matrix_collection->original_cm; };
    bool is_traction_discontinuity_enabled() { return matrix_collection->enable_traction_discontinuity; };
    Eigen::Index get_original_num_vertices() { return matrix_collection->original_num_vertices; };
    Eigen::Index get_num_vertices() { return matrix_collection->num_vertices; };

    const MatrixX3s &get_V_estimate_global() const { return V_estimate_global; };
    const MatrixX3s &get_V_global() const { return V_global; };
    VectorXs get_normalized_p() const {
        VectorXs temp = Eigen::Map<const Eigen::Matrix<ScalarType, -1, -1, Eigen::RowMajor>>(p.data(), p.rows() / 3, 3)
                            .rowwise()
                            .norm();
        ScalarType max = temp.maxCoeff();
        return max == 0.0 ? temp : temp / max;
    };
    const Matrix3s &get_rotation() const { return rotation; };
    const Vector3s &get_translation() const { return translation; };

    const Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> &get_permutation_matrix() const {
        return matrix_collection->perm;
    };

    virtual void estimate_next_state() = 0;
    virtual void confirm_next_state() = 0;

    ScalarType dt;
    Vector3s gravitational_constant;
    bool is_fixed;
    bool is_static;
    bool is_deformable;
    bool has_constraints;
    bool wireframe;
    std::vector<Eigen::Index> fixed_vertices;

  protected:
    Matrix3s rotational_vec_to_rotation_mat(const Vector3s &rotation_vec);
    virtual void update_V_estimate_matrices();
    virtual void update_conversion_matrices(
        const std::vector<Eigen::Index> &c1_vertex_indices, const MatrixX3s &c1_normal_vectors,
        const std::vector<Eigen::Index> &c2_face_indices, const MatrixX3s &c2_normal_vectors,
        const MatrixX3s &c2_barycentric_coords, const Matrix3s &o2_rotation, SparseMatrix &f1_f1c, SparseMatrix &f1_f2c,
        SparseMatrix &f1c_f1N, SparseMatrix &f1c_f1F, SparseMatrix &N1_V1, SparseMatrix &N2_V1, SparseMatrix &F1_V1,
        SparseMatrix &F2_V1
    );

    const std::shared_ptr<const PhysicsBaseMatrixCollection> matrix_collection;

    bool use_gravity;

    MatrixX3s V_local;
    Matrix3s rotation;
    Vector3s translation, translational_vel, rotational_vel;

    MatrixX3s V_estimate_global, V_global;

    VectorXs p;
    VectorXs V_const;
    DiagonalMatrixXs p_f;
    SparseMatrix V_u;
    MatrixX3s rot_trans_acc, rot_rot_acc;

    ScalarType friction_coeff;
    FrictionCombine friction_combine;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    friend ConstraintsSolver;
};

#endif //!__PHYSICS_BASE_OBJECT_HPP__