#ifndef __COLLISION_DETECTOR__
#define __COLLISION_DETECTOR__

#include <memory>
#include <unordered_map>
#include <vector>

#include "json.hpp"
#include "type_declaration.hpp"
#include "wavelets.hpp"

class PhysicsBaseObject;

namespace Geometry {
Vector3s closest_point_line(const Vector3s &p1, const Vector3s &p2, const Vector3s &q);
Vector3s closest_point_plane(const Vector3s &p, const Vector3s &n, const Vector3s &q);
Vector3s closest_line_plane(const Vector3s &p1, const Vector3s &p2, const Vector3s &q, const Vector3s &n);
bool point_in_triangle(const Vector3s &p1, const Vector3s &p2, const Vector3s &p3, const Vector3s &q);
bool triangle_sphere_intersection(
    const Vector3s &p1, const Vector3s &p2, const Vector3s &p3, const Vector3s &q, ScalarType radius
);
Vector3i point_to_grid(const Vector3s &p, ScalarType grid_interval);

class AABB {
  public:
    AABB();
    // from triangle
    AABB(const Vector3s &p1, const Vector3s &p2, const Vector3s &p3);
    // from mesh
    AABB(const MatrixX3s &V);
    // from sphere
    AABB(const Vector3s &p, ScalarType radius);

    bool intersects(const AABB &other) const;

    Vector3s p_min;
    Vector3s p_max;
};
} // namespace Geometry

class ConstraintsSolver {
  public:
    ConstraintsSolver();
    ~ConstraintsSolver();

    bool init(const nlohmann::json &config, std::vector<std::unique_ptr<PhysicsBaseObject>> &physics_objects);
    bool update_meshes();
    bool detect_collision();
    void precompute();
    void solve_constraints();

  private:
    std::tuple<long, long, long> point_to_grid(const Vector3s &p);
    void init_glue_constraints();
    void precompute_collision_constraints();
    void precompute_glue_constraints();
    void precompute_fixed_pos_constraints();

    template <class T> struct PAIR_HASH {
        size_t operator()(const T &pair) const {
            auto [a, b] = pair;
            return std::hash<decltype(a)>()(a) ^ std::hash<decltype(b)>()(b);
        }
    };

    class PhysicsObjectWrapper {
      public:
        struct Vector3i_HASH {
            size_t operator()(const Vector3i &val) const {
                return std::hash<IntType>()(val(0)) ^ std::hash<IntType>()(val(1)) ^ std::hash<IntType>()(val(2));
            }
        };

        PhysicsObjectWrapper(PhysicsBaseObject &physics_object, ScalarType grid_interval, bool enable_self_collision);

        void update_triangle_spatial_map(ScalarType grid_interval);
        void init_self_collision_matrix();

        PhysicsBaseObject &physics_object;
        Geometry::AABB aabb;
        std::unordered_map<Vector3i, std::vector<Eigen::Index>, Vector3i_HASH> triangle_spatial_map;
        MatrixXb self_collision_matrix; // initialized only when self collision is enabled.
        VectorXs f;
        MatrixX3s VN;
        VectorXs V_init; // initialized only when positional constraints are there.
    };

    std::vector<PhysicsObjectWrapper> physics_object_wrappers;

    enum ConstraintType { COLLISION, GLUE, FIXED_POS };

    struct CollisionConstraintData {
        const ScalarType friction_coeff;

        const bool is_deformable1, is_deformable2;
        const bool use_compressed_matrix1, use_compressed_matrix2;

        const SparseMatrix f1_fN, f2_fN, f1_fF, f2_fF, p1_fN, p2_fN, p1_fF, p2_fF;

        const VectorXs A_NN_diag, A_FF_diag;
        const MatrixXs N_fN, F_fF, N_p1, N_p2, F_p1, F_p2;

        const SparseMatrix N_u1_comp, p1_comp_fN, N_u2_comp, p2_comp_fN, F_u1_comp, p1_comp_fF, F_u2_comp, p2_comp_fF;

        const VectorXi N_u1_cols, N_u2_cols, F_u1_cols, F_u2_cols;

        const DiagonalMatrixXs &p1_f1;
        const CompressedMatrix<> &u1_p1;
        const MatrixXs &u1_p1_uncompressed;
        const PermutationMatrix &comp_perm1;

        const DiagonalMatrixXs &p2_f2;
        const CompressedMatrix<> &u2_p2;
        const MatrixXs &u2_p2_uncompressed;
        const PermutationMatrix &comp_perm2;

        const VectorXs b_N, b_F;

        VectorXs fN, fF;
    };

    struct GlueConstraintData {
        const bool is_deformable1, is_deformable2;
        const bool use_compressed_matrix1, use_compressed_matrix2;

        const SparseMatrix f1_fP, f2_fP;

        const VectorXs A_PP_diag;
        const MatrixXs P_fP, P_p1, P_p2;
        const SparseMatrix P_u1_comp, p1_comp_fP, P_u2_comp, p2_comp_fP;

        const VectorXi P_u1_cols, P_u2_cols;

        const DiagonalMatrixXs &p_f1, &p_f2;
        const CompressedMatrix<> &u1_p1, &u2_p2;
        const MatrixXs &u1_p1_uncompressed, &u2_p2_uncompressed;
        const PermutationMatrix &comp_perm1, &comp_perm2;

        const VectorXs b;

        VectorXs fP;
    };

    struct FixedPosConstraintData {
        const bool is_deformable;
        const bool use_compressed_matrix;

        const SparseMatrix f_fP;

        const VectorXs A_PP_diag;
        const MatrixXs P_fP, P_p;
        const SparseMatrix P_u_comp, p_comp_fP;

        const VectorXi P_u_cols;

        const DiagonalMatrixXs &p_f;
        const CompressedMatrix<> &u_p;
        const MatrixXs &u_p_uncompressed;
        const PermutationMatrix &comp_perm;

        const VectorXs b;

        VectorXs fP;
    };

    struct ConstraintData {
        ConstraintType constraint_type;
        bool is_single_object_contraint;
        size_t object1_idx;
        size_t object2_idx;

        // Either one of these is initialized. I do not use polymorphism so I can use bracket initializer for aggregate
        // types.
        std::unique_ptr<CollisionConstraintData> collision_constraint_data;
        std::unique_ptr<GlueConstraintData> glue_constraint_data;
        std::unique_ptr<FixedPosConstraintData> fixed_pos_constraint_data;
    };

    bool solve_collision_constraints(ConstraintData &);
    bool solve_glue_constraints(ConstraintData &);
    bool solve_fixed_pos_constraints(ConstraintData &);

    std::unordered_map<
        std::pair<size_t, size_t>, std::vector<std::pair<Eigen::Index, std::pair<Eigen::Index, RowVector3s>>>,
        PAIR_HASH<std::pair<size_t, size_t>>>
        collision_constraints;

    std::vector<std::tuple<size_t, size_t, const std::vector<Eigen::Index>, const std::vector<Eigen::Index>>>
        glue_constraints;

    std::vector<size_t> position_constraints;

    std::vector<ConstraintData> constraints_data_vector;
    std::vector<std::vector<std::reference_wrapper<ConstraintData>>> constraints_solution_batches;

    ScalarType epsilon, collision_radius, grid_interval, diagonal_scaling_factor;
    unsigned int global_iter, normal_iter, tangential_iter, glue_iter, fixed_pos_iter;
    bool enable_collision, enable_self_collision, enable_glue, enable_fixed_pos;
};

#endif //__COLLISION_DETECTOR__