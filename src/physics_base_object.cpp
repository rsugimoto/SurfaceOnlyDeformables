#include "physics_base_object.hpp"
#include "json.hpp"

#include <Eigen/Dense>
#include <igl/doublearea.h>

PhysicsBaseObject::PhysicsBaseObject(
    const std::shared_ptr<const PhysicsBaseMatrixCollection> matrix_collection, ScalarType dt,
    const Vector3s &gravitational_constant
)
    : dt(dt), gravitational_constant(gravitational_constant), is_fixed(false), wireframe(false),
      matrix_collection(matrix_collection), use_gravity(true), V_local(matrix_collection->get_V()),
      rotation(Matrix3s::Identity()), translation(Vector3s::Zero()), translational_vel(Vector3s::Zero()),
      rotational_vel(Vector3s::Zero()), friction_coeff(0.0), friction_combine(Average) {}

bool PhysicsBaseObject::init(const nlohmann::json &config) {
    if (config.contains("translational_vel")) {
        for (int i = 0; i < 3; i++) translational_vel(i) = config["translational_vel"][i].get<ScalarType>();
    }

    if (config.contains("rotational_vel")) {
        for (int i = 0; i < 3; i++) rotational_vel(i) = config["rotational_vel"][i].get<ScalarType>();
    }

    if (config.contains("rotation")) {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) rotation(i, j) = config["rotation"][3 * i + j].get<ScalarType>();
    }

    if (config.contains("translation")) {
        for (int i = 0; i < 3; i++) translation(i) = config["translation"][i].get<ScalarType>();

        translation += rotation * (matrix_collection->original_cm.transpose());
    }

    V_global = ((V_local.rowwise() - matrix_collection->cm) * rotation.transpose()).rowwise() +
               (translation.transpose() + matrix_collection->cm);
    V_estimate_global = V_global;

    if (config.contains("use_gravity")) use_gravity = config["use_gravity"].get<bool>();

    if (config.contains("is_fixed")) is_fixed = config["is_fixed"].get<bool>();

    if (config.contains("friction_coeff")) friction_coeff = config["friction_coeff"].get<ScalarType>();

    if (config.contains("wireframe")) wireframe = config["wireframe"].get<bool>();

    if (config.contains("friction_combine")) {
        if (config["friction_combine"].get<std::string>() == "maximum")
            friction_combine = Maximum;
        else if (config["friction_combine"].get<std::string>() == "multiply")
            friction_combine = Multiply;
        else if (config["friction_combine"].get<std::string>() == "minimum")
            friction_combine = Minimum;
        else if (config["friction_combine"].get<std::string>() == "average")
            friction_combine = Average;
        else
            return false;
    }

    if (config.contains("fixed_vertices")) {
        bool add_range_flag = false;
        int prev_vert;
        for (auto vert : config["fixed_vertices"]) {
            try {
                int curr_vert = vert.get<int>();
                if (add_range_flag) {
                    for (int v = prev_vert + 1; v <= curr_vert; v++) fixed_vertices.emplace_back(v);
                    add_range_flag = false;
                } else {
                    fixed_vertices.emplace_back(curr_vert);
                    prev_vert = curr_vert;
                }
            } catch (...) { add_range_flag = true; }
        }
    }

    rot_trans_acc = MatrixX3s::Zero(3 * matrix_collection->num_vertices, 3);
    rot_rot_acc = MatrixX3s::Zero(3 * matrix_collection->num_vertices, 3);
    V_u.resize(3 * matrix_collection->num_vertices, 3 * matrix_collection->num_vertices);

    return true;
}

Matrix3s PhysicsBaseObject::rotational_vec_to_rotation_mat(const Vector3s &rotation_vec) {
    const Vector3s rotation_vec_normalized = rotation_vec.normalized();
    const ScalarType rotation_vec_norm = rotation_vec.stableNorm();
    Matrix3s rot_vec_skew_mat;
    rot_vec_skew_mat << 0.0, -rotation_vec_normalized(2), rotation_vec_normalized(1), rotation_vec_normalized(2), 0.0,
        -rotation_vec_normalized(0), -rotation_vec_normalized(1), rotation_vec_normalized(0), 0.0;
    // Rodrigues' rotation formula
    Matrix3s exp_rotation_vec = Matrix3s::Identity() + rot_vec_skew_mat * std::sin(rotation_vec_norm) +
                                rot_vec_skew_mat * rot_vec_skew_mat * (1. - std::cos(rotation_vec_norm));
    return exp_rotation_vec;
}

void PhysicsBaseObject::update_V_estimate_matrices() {
    // SparseMatrix p_f;
    {
        VectorXs A;
        igl::doublearea(V_local, matrix_collection->F, A);

        VectorXs vertex_area = VectorXs::Zero(V_local.rows() * 3);
        for (Eigen::Index f = 0; f < matrix_collection->F.rows(); f++) {
            for (int v = 0; v < 3; v++) {
                vertex_area(3 * matrix_collection->F(f, v)) += A(f) / 6.0;
                vertex_area(3 * matrix_collection->F(f, v) + 1) += A(f) / 6.0;
                vertex_area(3 * matrix_collection->F(f, v) + 2) += A(f) / 6.0;
            }
        }

        p_f = vertex_area.cwiseInverse().asDiagonal();
    }
}

void PhysicsBaseObject::update_conversion_matrices(
    const std::vector<Eigen::Index> &c1_vertex_indices, const MatrixX3s &c1_normal_vectors,
    const std::vector<Eigen::Index> &c2_face_indices, const MatrixX3s &c2_normal_vectors,
    const MatrixX3s &c2_barycentric_coords, const Matrix3s &o2_rotation, SparseMatrix &f1_f1c, SparseMatrix &f1_f2c,
    SparseMatrix &f1c_f1N, SparseMatrix &f1c_f1F, SparseMatrix &N1_V1, SparseMatrix &N2_V1, SparseMatrix &F1_V1,
    SparseMatrix &F2_V1
) {
    size_t num_collision_vertices = c1_vertex_indices.size();

    // SparseMatrix f1_f1c;
    {
        std::vector<Eigen::Triplet<ScalarType>> elems;
        elems.reserve(num_collision_vertices * 3);
        Eigen::Index v_new = 0;
        for (auto v : c1_vertex_indices) {
            elems.emplace_back(3 * v, 3 * v_new, 1.0);
            elems.emplace_back(3 * v + 1, 3 * v_new + 1, 1.0);
            elems.emplace_back(3 * v + 2, 3 * v_new + 2, 1.0);
            v_new++;
        }
        f1_f1c.resize(matrix_collection->num_vertices * 3, num_collision_vertices * 3);
        f1_f1c.setFromTriplets(elems.begin(), elems.end());
    }

    // SparseMatrix f1_f2c;
    {
        std::vector<Eigen::Triplet<ScalarType>> elems;
        elems.reserve(num_collision_vertices * 3 * 3 * 3);
        Eigen::Index v_new = 0;
        Matrix3s _rotation = rotation.transpose() * o2_rotation;
        for (auto f : c2_face_indices) {
            auto face = matrix_collection->F.row(f);
            auto bc = c2_barycentric_coords.row(v_new);
            for (int v = 0; v < 3; v++) {
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        elems.emplace_back(3 * face(v) + i, 3 * v_new + j, -_rotation(i, j) * bc(v));
                    }
                }
            }
            v_new++;
        }
        f1_f2c.resize(matrix_collection->num_vertices * 3, c2_face_indices.size() * 3);
        f1_f2c.setFromTriplets(elems.begin(), elems.end());
    }

    // SparseMatrix f1c_f1N;
    {
        std::vector<Eigen::Triplet<ScalarType>> triplets;
        triplets.reserve(num_collision_vertices * 3);
        for (Eigen::Index v = 0; v < num_collision_vertices; v++) {
            auto n = c1_normal_vectors.row(v) * rotation;
            triplets.emplace_back(3 * v, v, n(0));
            triplets.emplace_back(3 * v + 1, v, n(1));
            triplets.emplace_back(3 * v + 2, v, n(2));
        }
        f1c_f1N.resize(num_collision_vertices * 3, num_collision_vertices);
        f1c_f1N.setFromTriplets(triplets.begin(), triplets.end());
    }

    auto get_orthogonal_unit_vector_mat = [&](const Vector3s &vec) {
        Eigen::Matrix<ScalarType, 3, 2> D;
        if (std::abs(vec.dot(Vector3s(1., 0., 0.))) < 0.9)
            D.col(0) = (vec.cross(Vector3s(1., 0., 0.))).normalized();
        else
            D.col(0) = (vec.cross(Vector3s(0., 1., 0.))).normalized();
        D.col(1) = (vec.cross(D.col(0))).normalized();
        return D;
    };

    // SparseMatrix f1c_f1F;
    {
        std::vector<Eigen::Triplet<ScalarType>> triplets;
        triplets.reserve(num_collision_vertices * 6);
        for (Eigen::Index v = 0; v < num_collision_vertices; v++) {
            auto mat = get_orthogonal_unit_vector_mat(c1_normal_vectors.row(v) * rotation);
            triplets.emplace_back(3 * v, 2 * v, mat(0, 0));
            triplets.emplace_back(3 * v + 1, 2 * v, mat(1, 0));
            triplets.emplace_back(3 * v + 2, 2 * v, mat(2, 0));
            triplets.emplace_back(3 * v, 2 * v + 1, mat(0, 1));
            triplets.emplace_back(3 * v + 1, 2 * v + 1, mat(1, 1));
            triplets.emplace_back(3 * v + 2, 2 * v + 1, mat(2, 1));
        }
        f1c_f1F.resize(num_collision_vertices * 3, num_collision_vertices * 2);
        f1c_f1F.setFromTriplets(triplets.begin(), triplets.end());
    }

    // SparseMatrix N1_V1;
    {
        std::vector<Eigen::Triplet<ScalarType>> elems;
        elems.reserve(num_collision_vertices * 3);
        Eigen::Index v_new = 0;
        for (auto v : c1_vertex_indices) {
            elems.emplace_back(v_new, 3 * v, c1_normal_vectors(v_new, 0));
            elems.emplace_back(v_new, 3 * v + 1, c1_normal_vectors(v_new, 1));
            elems.emplace_back(v_new, 3 * v + 2, c1_normal_vectors(v_new, 2));
            v_new++;
        }
        N1_V1.resize(num_collision_vertices, matrix_collection->num_vertices * 3);
        N1_V1.setFromTriplets(elems.begin(), elems.end());
    }

    // SparseMatrix N2_V1;
    {
        std::vector<Eigen::Triplet<ScalarType>> elems;
        elems.reserve(num_collision_vertices * 3 * 3);
        Eigen::Index v_new = 0;
        for (auto f : c2_face_indices) {
            auto face = matrix_collection->F.row(f);
            auto bc = c2_barycentric_coords.row(v_new);
            auto n = c2_normal_vectors.row(v_new);
            for (int v = 0; v < 3; v++) {
                elems.emplace_back(v_new, 3 * face(v), n(0) * bc(v));
                elems.emplace_back(v_new, 3 * face(v) + 1, n(1) * bc(v));
                elems.emplace_back(v_new, 3 * face(v) + 2, n(2) * bc(v));
            }
            v_new++;
        }
        N2_V1.resize(c2_face_indices.size(), matrix_collection->num_vertices * 3);
        N2_V1.setFromTriplets(elems.begin(), elems.end());
    }

    // SparseMatrix F1_V1;
    {
        std::vector<Eigen::Triplet<ScalarType>> elems;
        elems.reserve(num_collision_vertices * 6);
        Eigen::Index v_new = 0;
        for (auto v : c1_vertex_indices) {
            auto mat = get_orthogonal_unit_vector_mat(c1_normal_vectors.row(v_new).transpose());
            elems.emplace_back(2 * v_new, 3 * v, mat(0, 0));
            elems.emplace_back(2 * v_new, 3 * v + 1, mat(1, 0));
            elems.emplace_back(2 * v_new, 3 * v + 2, mat(2, 0));
            elems.emplace_back(2 * v_new + 1, 3 * v, mat(0, 1));
            elems.emplace_back(2 * v_new + 1, 3 * v + 1, mat(1, 1));
            elems.emplace_back(2 * v_new + 1, 3 * v + 2, mat(2, 1));
            v_new++;
        }
        F1_V1.resize(num_collision_vertices * 2, matrix_collection->num_vertices * 3);
        F1_V1.setFromTriplets(elems.begin(), elems.end());
    }

    // SparseMatrix F2_V1;
    {
        std::vector<Eigen::Triplet<ScalarType>> elems;
        elems.reserve(num_collision_vertices * 6 * 3);
        Eigen::Index v_new = 0;
        for (auto f : c2_face_indices) {
            auto face = matrix_collection->F.row(f);
            auto bc = c2_barycentric_coords.row(v_new);
            auto mat = get_orthogonal_unit_vector_mat(c2_normal_vectors.row(v_new).transpose());
            for (int v = 0; v < 3; v++) {
                elems.emplace_back(2 * v_new, 3 * face(v), mat(0, 0) * bc(v));
                elems.emplace_back(2 * v_new, 3 * face(v) + 1, mat(1, 0) * bc(v));
                elems.emplace_back(2 * v_new, 3 * face(v) + 2, mat(2, 0) * bc(v));
                elems.emplace_back(2 * v_new + 1, 3 * face(v), mat(0, 1) * bc(v));
                elems.emplace_back(2 * v_new + 1, 3 * face(v) + 1, mat(1, 1) * bc(v));
                elems.emplace_back(2 * v_new + 1, 3 * face(v) + 2, mat(2, 1) * bc(v));
            }
            v_new++;
        }
        F2_V1.resize(c2_face_indices.size() * 2, matrix_collection->num_vertices * 3);
        F2_V1.setFromTriplets(elems.begin(), elems.end());
    }
}