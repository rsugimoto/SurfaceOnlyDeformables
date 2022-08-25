#include "elastostatics_bem_object.hpp"
#include "elastostatics_bem_matrix_collection.hpp"

#include "cross_product_matrix.hpp"

ElastostaticsBEMObject::ElastostaticsBEMObject(
    const std::shared_ptr<const ElastostaticsBEMMatrixCollection> matrix_collection, ScalarType dt,
    const Vector3s &gravitational_constant
)
    : PhysicsBaseObject(matrix_collection, dt, gravitational_constant) {}

ElastostaticsBEMObject::~ElastostaticsBEMObject() {}

bool ElastostaticsBEMObject::init(const nlohmann::json &config) {
    if (!PhysicsBaseObject::init(config)) return false;
    is_static = true;
    is_deformable = true;

    return true;
}

void ElastostaticsBEMObject::estimate_next_state() {
    has_constraints = false;
    const ElastostaticsBEMMatrixCollection *const matrix_collection =
        static_cast<const ElastostaticsBEMMatrixCollection *>(this->matrix_collection.get());

    p = VectorXs::Zero(matrix_collection->num_vertices * 3);
    if (!is_fixed) {
        Matrix3s rotation_estimate = rotational_vec_to_rotation_mat(dt * rotational_vel) * rotation;
        Vector3s translation_estimate = translation + dt * translational_vel;
        if (use_gravity) translation_estimate += dt * dt * gravitational_constant;

        V_estimate_global = matrix_collection->V * rotation_estimate.transpose();
        V_estimate_global.rowwise() += translation_estimate.transpose();
    }
}

void ElastostaticsBEMObject::update_V_estimate_matrices() {
    PhysicsBaseObject::update_V_estimate_matrices();

    const ElastostaticsBEMMatrixCollection *const matrix_collection =
        static_cast<const ElastostaticsBEMMatrixCollection *>(this->matrix_collection.get());

    // Assumes centroid is (0, 0, 0)
    const MatrixX3s &V_vec = matrix_collection->get_V();

    {
        Matrix3s rotation_estimate = rotational_vec_to_rotation_mat(dt * rotational_vel) * rotation;
        std::vector<Eigen::Triplet<ScalarType>> elems;
        elems.reserve(3 * 3 * matrix_collection->num_vertices);
        for (Eigen::Index v = 0; v < matrix_collection->num_vertices; v++) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) { elems.emplace_back(3 * v + i, 3 * v + j, rotation_estimate(i, j)); }
            }
        }
        V_u.setFromTriplets(elems.begin(), elems.end());
    }

    if (!is_fixed && matrix_collection->update_local_frame) {
        Matrix3s dt_dt_rot = dt * dt * rotation;
        for (Eigen::Index v = 0; v < matrix_collection->num_vertices; v++) {
            rot_trans_acc.middleRows<3>(3 * v) = dt_dt_rot;
            rot_rot_acc.middleRows<3>(3 * v) = -dt_dt_rot * cross_product_matrix(V_vec.row(v));
        }
    }

    V_const.resize(matrix_collection->num_vertices * 3);
    auto V_const_mat = Eigen::Map<Eigen::Matrix<ScalarType, -1, 3, Eigen::RowMajor>>(
        V_const.data(), matrix_collection->num_vertices, 3
    );
    V_const_mat = V_estimate_global;
    // Cancel out the terms when necessary
    if (!is_fixed && !matrix_collection->update_local_frame) {
        V_const_mat.rowwise() -= dt * translational_vel.transpose();
        if (use_gravity) V_const_mat.rowwise() -= dt * dt * gravitational_constant.transpose();
    }
}

void ElastostaticsBEMObject::confirm_next_state() {
    const ElastostaticsBEMMatrixCollection *const matrix_collection =
        static_cast<const ElastostaticsBEMMatrixCollection *>(this->matrix_collection.get());

    VectorXs u;
    if (matrix_collection->use_compressed_matrices)
        u = matrix_collection->compression_permutation.transpose() *
            (VectorXs)(matrix_collection->u_p_compressed * (matrix_collection->compression_permutation * p));
    else
        u = matrix_collection->u_p * p;

    V_local =
        matrix_collection->get_V() +
        Eigen::Map<Eigen::Matrix<ScalarType, -1, 3, Eigen::RowMajor>>(u.data(), matrix_collection->num_vertices, 3);

    // Compute the new frame values
    if (!is_fixed && matrix_collection->update_local_frame) {
        Vector3s new_translational_acc = matrix_collection->translational_acc_mat * p;
        if (use_gravity) new_translational_acc += rotation.transpose() * gravitational_constant;
        Vector3s new_rotational_acc = matrix_collection->rotational_acc_mat * p;
        translational_vel = translational_vel + dt * rotation * new_translational_acc;
        rotational_vel = rotational_vel + dt * rotation * new_rotational_acc;
        translation = translation + dt * translational_vel;
        rotation = rotational_vec_to_rotation_mat(dt * rotational_vel) * rotation;
    }
    V_global = (V_local * rotation.transpose()).rowwise() + translation.transpose();
}