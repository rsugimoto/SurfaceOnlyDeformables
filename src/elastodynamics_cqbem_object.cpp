#include "elastodynamics_cqbem_object.hpp"
#include "elastodynamics_cqbem_matrix_collection.hpp"

#include "cross_product_matrix.hpp"
#include <Eigen/SVD>
#include <igl/centroid.h>
#include <iostream>

ElastodynamicsCQBEMObject::ElastodynamicsCQBEMObject(
    const std::shared_ptr<const ElastodynamicsCQBEMMatrixCollection> matrix_collection, ScalarType dt,
    const Vector3s &gravitational_constant
)
    : PhysicsBaseObject(matrix_collection, dt, gravitational_constant) {}

ElastodynamicsCQBEMObject::~ElastodynamicsCQBEMObject() {}

bool ElastodynamicsCQBEMObject::init(const nlohmann::json &config) {
    if (!PhysicsBaseObject::init(config)) return false;
    is_static = false;
    is_deformable = true;

    return true;
}

void ElastodynamicsCQBEMObject::estimate_next_state() {
    has_constraints = false;
    const ElastodynamicsCQBEMMatrixCollection *const matrix_collection =
        static_cast<const ElastodynamicsCQBEMMatrixCollection *>(this->matrix_collection.get());

    if (matrix_collection->use_compressed_matrices) {
        b = VectorXs::Zero(matrix_collection->num_vertices * 3);
        CompressedMatrixVectorProduct<> b_compressed =
            CompressedMatrixVectorProduct<>(matrix_collection->num_vertices * 3);
        for (size_t i = 0; i < std::min(hist_acc_vectors.size(), matrix_collection->max_time_history); i++) {
            b += matrix_collection->B_matrices[i + 1] * hist_acc_vectors[hist_acc_vectors.size() - i - 1];
            b_compressed += matrix_collection->H_compressed_matrices[i + 1] *
                            hist_u_compressed_vectors[hist_acc_vectors.size() - i - 1];
            b_compressed += matrix_collection->G_compressed_matrices[i + 1] *
                            hist_p_compressed_vectors[hist_acc_vectors.size() - i - 1];
        }
        b += (matrix_collection->compression_permutation.transpose() * (VectorXs)b_compressed).eval();
    } else {
        b = VectorXs::Zero(matrix_collection->num_vertices * 3);
        for (size_t i = 0; i < std::min(hist_acc_vectors.size(), matrix_collection->max_time_history); i++) {
            b += matrix_collection->B_matrices[i + 1] * hist_acc_vectors[hist_acc_vectors.size() - i - 1];
            b += matrix_collection->H_matrices[i + 1] * hist_u_vectors[hist_acc_vectors.size() - i - 1];
            b += matrix_collection->G_matrices[i + 1] * hist_p_vectors[hist_acc_vectors.size() - i - 1];
        }
    }

    // without local frame update, the gravitational force term and translational fictitious force term do not cancel
    // out with each other.
    if (!matrix_collection->update_local_frame && use_gravity)
        b -= matrix_collection->B_matrices[0].leftCols<3>() * rotation.transpose() * gravitational_constant;

    p = VectorXs::Zero(matrix_collection->num_vertices * 3);

    if (matrix_collection->use_compressed_matrices)
        u_unconstrained =
            matrix_collection->compression_permutation.transpose() *
            (VectorXs)(matrix_collection->H_inv_compressed * (matrix_collection->compression_permutation * b));
    else
        u_unconstrained = matrix_collection->H_inv.solve(b);

    if (is_fixed) {
        V_estimate_global = (matrix_collection->V + Eigen::Map<Eigen::Matrix<ScalarType, -1, 3, Eigen::RowMajor>>(
                                                        u_unconstrained.data(), matrix_collection->num_vertices, 3
                                                    )) *
                            rotation.transpose();
        V_estimate_global.rowwise() += translation.transpose();
    } else {
        Matrix3s rotation_estimate = rotational_vec_to_rotation_mat(dt * rotational_vel) * rotation;
        Vector3s translation_estimate = translation + dt * translational_vel;
        if (use_gravity) translation_estimate += dt * dt * gravitational_constant;

        V_estimate_global = (matrix_collection->V + Eigen::Map<Eigen::Matrix<ScalarType, -1, 3, Eigen::RowMajor>>(
                                                        u_unconstrained.data(), matrix_collection->num_vertices, 3
                                                    )) *
                            rotation_estimate.transpose();
        V_estimate_global.rowwise() += translation_estimate.transpose();
    }
}

void ElastodynamicsCQBEMObject::update_V_estimate_matrices() {
    PhysicsBaseObject::update_V_estimate_matrices();

    const ElastodynamicsCQBEMMatrixCollection *const matrix_collection =
        static_cast<const ElastodynamicsCQBEMMatrixCollection *>(this->matrix_collection.get());

    // Assumes centroid is (0, 0, 0)
    const MatrixX3s &V_vec = V_local;

    {
        Matrix3s rotation_estimate =
            is_fixed ? rotation : rotational_vec_to_rotation_mat(dt * rotational_vel) * rotation;
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

void ElastodynamicsCQBEMObject::confirm_next_state() {
    const ElastodynamicsCQBEMMatrixCollection *const matrix_collection =
        static_cast<const ElastodynamicsCQBEMMatrixCollection *>(this->matrix_collection.get());

    VectorXs u, u_p;
    if (has_constraints) {
        if (matrix_collection->use_compressed_matrices)
            u_p = matrix_collection->compression_permutation.transpose() *
                  (VectorXs)(matrix_collection->u_p_compressed * (matrix_collection->compression_permutation * p));
        else
            u_p = matrix_collection->u_p * p;
        u = u_p + u_unconstrained;
        // if constraint solver is applied, use the result of constraint solve as the vertex positions in the next step.
        Eigen::Map<VectorXs> V_global_vector = Eigen::Map<VectorXs>(V_global.data(), V_global.size());
        V_global_vector = V_u * u_p + V_const;
        if (!is_fixed && matrix_collection->update_local_frame)
            V_global_vector += rot_trans_acc * (matrix_collection->translational_acc_mat * p) +
                               rot_rot_acc * (matrix_collection->rotational_acc_mat * p);
    } else {
        u = u_unconstrained;
        u_p = VectorXs::Zero(matrix_collection->num_vertices * 3);
    }
    V_local =
        matrix_collection->get_V() +
        Eigen::Map<Eigen::Matrix<ScalarType, -1, 3, Eigen::RowMajor>>(u.data(), matrix_collection->num_vertices, 3);

    Vector3s _translational_acc = Vector3s::Zero(), _rotational_acc = Vector3s::Zero();
    if (matrix_collection->update_local_frame) {
        Matrix3s old_rotation = rotation;

        // Compute the new frame values
        {
            _translational_acc = matrix_collection->translational_acc_mat * p;
            _rotational_acc = matrix_collection->rotational_acc_mat * p;
            if (!is_fixed) {
                translational_vel = translational_vel + dt * rotation * _translational_acc;
                if (use_gravity) translational_vel += dt * gravitational_constant;
                rotational_vel = rotational_vel + dt * rotation * _rotational_acc;
                translation = translation + dt * translational_vel;
                rotation = rotational_vec_to_rotation_mat(dt * rotational_vel) * rotation;
            }
        }

        if (!has_constraints) V_global = (V_local * rotation.transpose()).rowwise() + translation.transpose();

        // Drift elimination
        {
            RowVector3s cm;
            igl::centroid(V_local, matrix_collection->F, cm);
            MatrixX3s V_vec = (V_local.rowwise() - cm).rowwise().normalized();
            MatrixX3s init_V_vec =
                (matrix_collection->get_V().rowwise() - matrix_collection->cm).rowwise().normalized();

            Matrix3s F = init_V_vec.transpose() * matrix_collection->rotation_removal_weights.asDiagonal() * V_vec;
            Eigen::JacobiSVD<Matrix3s> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Matrix3s R = svd.matrixV() * (svd.matrixU().transpose());

            auto u_mat = Eigen::Map<Eigen::Matrix<ScalarType, -1, 3, Eigen::RowMajor>>(
                u.data(), matrix_collection->num_vertices, 3
            );

            u_mat = (V_local.rowwise() - cm) - (matrix_collection->get_V().rowwise() - matrix_collection->cm);
            Vector3s translational_acc_delta = matrix_collection->B_trans_inv_H * u -
                                               matrix_collection->B_trans_inv_G_prime * p -
                                               matrix_collection->B_trans_inv.solve(b);
            u_mat = (V_local.rowwise() - cm) * R - (matrix_collection->get_V().rowwise() - matrix_collection->cm);
            Vector3s angular_acc_delta = matrix_collection->B_euler_inv_H * u -
                                         matrix_collection->B_euler_inv_G_prime * p -
                                         matrix_collection->B_euler_inv.solve(b) -
                                         matrix_collection->B_euler_inv_B_trans * translational_acc_delta;
            u = u_p + u_unconstrained + matrix_collection->H_inv_B_trans * translational_acc_delta +
                matrix_collection->H_inv_B_euler * angular_acc_delta;

            _translational_acc += translational_acc_delta;
            _rotational_acc += angular_acc_delta;

            if (!is_fixed) {
                translational_vel += dt * old_rotation * translational_acc_delta;
                rotational_vel += dt * old_rotation * angular_acc_delta;
                translation = translation + dt * dt * old_rotation * translational_acc_delta;
                rotation = rotational_vec_to_rotation_mat(dt * rotational_vel) * old_rotation;
            }
            V_local = matrix_collection->get_V() + Eigen::Map<Eigen::Matrix<ScalarType, -1, 3, Eigen::RowMajor>>(
                                                       u.data(), matrix_collection->num_vertices, 3
                                                   );
        }
    } else {
        if (use_gravity) _translational_acc = -rotation.transpose() * gravitational_constant;
        if (!has_constraints) V_global = (V_local * rotation.transpose()).rowwise() + translation.transpose();
    }

    if (matrix_collection->use_compressed_matrices) {
        if (hist_acc_vectors.size() >= matrix_collection->max_time_history) {
            hist_acc_vectors.pop_front();
            hist_u_compressed_vectors.pop_front();
            hist_p_compressed_vectors.pop_front();
        }
        hist_acc_vectors.emplace_back();
        hist_acc_vectors.back() << _translational_acc, _rotational_acc;
        hist_u_compressed_vectors.emplace_back(matrix_collection->compression_permutation * u);
        hist_p_compressed_vectors.emplace_back(matrix_collection->compression_permutation * p);
    } else {
        if (hist_acc_vectors.size() >= matrix_collection->max_time_history) {
            hist_acc_vectors.pop_front();
            hist_u_vectors.pop_front();
            hist_p_vectors.pop_front();
        }
        hist_acc_vectors.emplace_back();
        hist_acc_vectors.back() << _translational_acc, _rotational_acc;
        hist_u_vectors.emplace_back(u);
        hist_p_vectors.emplace_back(p);
    }
}
