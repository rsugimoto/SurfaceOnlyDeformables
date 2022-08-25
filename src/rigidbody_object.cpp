#include "rigidbody_object.hpp"
#include "rigidbody_matrix_collection.hpp"

#include "cross_product_matrix.hpp"

RigidbodyObject::RigidbodyObject(
    const std::shared_ptr<const RigidbodyMatrixCollection> matrix_collection, ScalarType dt,
    const Vector3s &gravitational_constant
)
    : PhysicsBaseObject(matrix_collection, dt, gravitational_constant){};

RigidbodyObject::~RigidbodyObject(){};

bool RigidbodyObject::init(const nlohmann::json &config) {
    if (!PhysicsBaseObject::init(config)) return false;
    is_static = true;
    is_deformable = false;
    return true;
}

void RigidbodyObject::estimate_next_state() {
    has_constraints = false;
    const RigidbodyMatrixCollection *const matrix_collection =
        static_cast<const RigidbodyMatrixCollection *>(this->matrix_collection.get());

    p = VectorXs::Zero(matrix_collection->num_vertices * 3);
    if (!is_fixed) {
        Matrix3s rotation_estimate = rotational_vec_to_rotation_mat(dt * rotational_vel) * rotation;
        Vector3s translation_estimate = translation + dt * translational_vel;
        if (use_gravity) translation_estimate += dt * dt * gravitational_constant;

        V_estimate_global = matrix_collection->V * rotation_estimate.transpose();
        V_estimate_global.rowwise() += translation_estimate.transpose();
    }
}

void RigidbodyObject::update_V_estimate_matrices() {
    PhysicsBaseObject::update_V_estimate_matrices();

    const RigidbodyMatrixCollection *const matrix_collection =
        static_cast<const RigidbodyMatrixCollection *>(this->matrix_collection.get());

    // Assumes centroid is (0, 0, 0)
    MatrixX3s &V_vec = V_local;

    if (!is_fixed) {
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
}

void RigidbodyObject::confirm_next_state() {
    if (is_fixed) return;

    const RigidbodyMatrixCollection *const matrix_collection =
        static_cast<const RigidbodyMatrixCollection *>(this->matrix_collection.get());

    // Compute the new frame values
    {
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
