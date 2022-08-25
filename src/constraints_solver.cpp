#include "constraints_solver.hpp"
#include "physics_base_matrix_collection.hpp"
#include "type_declaration.hpp"

#include <iostream>
#include <set>
#include <unordered_set>

#ifdef OMP_AVAILABLE
#include "omp.h"
#endif

#include <igl/barycentric_coordinates.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>

#include "physics_base_object.hpp"

namespace Geometry {
// reference
// https://github.com/gszauer/GamePhysicsCookbook/blob/master/Code/Geometry3D.cpp

inline Vector3s closest_point_line(const Vector3s &p1, const Vector3s &p2, const Vector3s &q) {
    Vector3s line_vec = p1 - p2;
    ScalarType t = (line_vec.dot(q - p2)) / (line_vec.squaredNorm());
    t = std::clamp(t, (ScalarType)0.0, (ScalarType)1.0);
    return p2 + t * line_vec;
}

inline Vector3s closest_point_plane(const Vector3s &p, const Vector3s &n, const Vector3s &q) {
    ScalarType signed_dist = (p - q).dot(n);
    return q + signed_dist * n;
}

// returns a closest point to a plane in a line
inline Vector3s closest_line_plane(const Vector3s &p1, const Vector3s &p2, const Vector3s &q, const Vector3s &n) {
    ScalarType denominator = n.dot(p2 - p1);
    ScalarType numerator = -n.dot(q - p1);
    ScalarType t = numerator / denominator;
    if (t < 0) return p1;
    if (t > 1) return p2;
    return p1 + t * (p2 - p1);
}

inline bool point_in_triangle(const Vector3s &p1, const Vector3s &p2, const Vector3s &p3, const Vector3s &q) {
    Vector3s _p1 = p1 - q, _p2 = p2 - q, _p3 = p3 - q;
    Vector3s n1 = _p2.cross(_p3), n2 = _p3.cross(_p1), n3 = _p1.cross(_p2);
    if (n1.dot(n2) < 0.0) return false;
    if (n2.dot(n3) < 0.0) return false;
    if (n3.dot(n1) < 0.0) return false;
    return true;
}

inline bool triangle_sphere_intersection(
    const Vector3s &p1, const Vector3s &p2, const Vector3s &p3, const Vector3s &q, ScalarType radius
) {
    Vector3s plane_closest = closest_point_plane(p1, ((p2 - p1).cross(p3 - p1)).normalized(), q);
    ScalarType squared_radius = radius * radius;
    if ((plane_closest - q).squaredNorm() > squared_radius) return false;
    if (point_in_triangle(p1, p2, p3, plane_closest)) return true;
    if ((closest_point_line(p1, p2, q) - q).squaredNorm() <= squared_radius) return true;
    if ((closest_point_line(p2, p3, q) - q).squaredNorm() <= squared_radius) return true;
    if ((closest_point_line(p3, p1, q) - q).squaredNorm() <= squared_radius) return true;
    return false;
}

inline Vector3i point_to_grid(const Vector3s &p, ScalarType grid_interval) {
    return (p / grid_interval).cast<IntType>();
}

AABB::AABB(const Vector3s &p1, const Vector3s &p2, const Vector3s &p3)
    : p_min(p1.cwiseMin(p2).cwiseMin(p3)), p_max(p1.cwiseMax(p2).cwiseMax(p3)) {}

AABB::AABB(const MatrixX3s &V) : p_min(V.colwise().minCoeff()), p_max(V.colwise().maxCoeff()) {}

AABB::AABB(const Vector3s &p, ScalarType radius) : p_min(p.array() - radius), p_max(p.array() + radius) {}

bool AABB::intersects(const AABB &other) const {
    return (p_min.array() <= other.p_max.array()).all() && (p_max.array() >= other.p_min.array()).all();
}
} // namespace Geometry

ConstraintsSolver::PhysicsObjectWrapper::PhysicsObjectWrapper(
    PhysicsBaseObject &physics_object, ScalarType grid_interval, bool enable_self_collision
)
    : physics_object(physics_object), aabb(Geometry::AABB(physics_object.get_V_estimate_global())) {

    update_triangle_spatial_map(grid_interval);

    if (enable_self_collision) init_self_collision_matrix();

    if (physics_object.fixed_vertices.size() > 0)
        V_init = Eigen::Map<const VectorXs>(
            physics_object.get_V_estimate_global().data(), physics_object.get_num_vertices() * 3
        );

    f = VectorXs::Zero(physics_object.get_num_vertices() * 3);
}

void ConstraintsSolver::PhysicsObjectWrapper::update_triangle_spatial_map(ScalarType grid_interval) {
    triangle_spatial_map.clear();
    const MatrixX3i &F = physics_object.get_F();
    const MatrixX3s &V = physics_object.get_V_estimate_global();

#pragma omp parallel for
    for (Eigen::Index f = 0; f < F.rows(); f++) {
        Geometry::AABB aabb(V.row(F(f, 0)), V.row(F(f, 1)), V.row(F(f, 2)));
        Vector3i min_coords = Geometry::point_to_grid(aabb.p_min, grid_interval);
        Vector3i max_coords = Geometry::point_to_grid(aabb.p_max, grid_interval);
        for (auto x = min_coords[0]; x <= max_coords[0]; x++) {
            for (auto y = min_coords[1]; y <= max_coords[1]; y++) {
                for (auto z = min_coords[2]; z <= max_coords[2]; z++) {
#pragma omp critical
                    {
                        Vector3i grid_coord(x, y, z);
                        triangle_spatial_map.try_emplace(grid_coord);
                        triangle_spatial_map[grid_coord].emplace_back(f);
                    }
                }
            }
        }
    }
}

void ConstraintsSolver::PhysicsObjectWrapper::init_self_collision_matrix() {
    const MatrixX3i &F = physics_object.get_F();
    const MatrixX3s &V = physics_object.get_V_estimate_global();

    self_collision_matrix = MatrixXb::Ones(V.rows(), F.rows());

    MatrixX3s VN;
    igl::per_vertex_normals(V, F, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, VN);

    MatrixX3s FN;
    igl::per_face_normals(V, F, FN);

    for (size_t v = 0; v < V.rows(); v++) {
        auto v_pows = V.row(v);
        auto vn = VN.row(v);
        for (size_t f = 0; f < F.rows(); f++) {
            if (v == F(f, 0) || v == F(f, 1) || v == F(f, 2)) {
                self_collision_matrix(v, f) = 0;
                continue;
            }

            RowVector3s f_pows = (V.row(F(f, 0)) + V.row(F(f, 1)) + V.row(F(f, 2))) / 3.;
            auto fn = FN.row(f);
            if (!(fn.dot(vn) < 0.0 && (f_pows - v_pows).dot(vn) > 0.0)) {
                self_collision_matrix(v, f) = 0;
                continue;
            }
        }
    }
}

ConstraintsSolver::ConstraintsSolver()
    : epsilon(1e-5), collision_radius(0.1), grid_interval(0.5), diagonal_scaling_factor(1e-4), global_iter(20),
      normal_iter(20), tangential_iter(10), glue_iter(20), fixed_pos_iter(20), enable_collision(true),
      enable_self_collision(false), enable_glue(false), enable_fixed_pos(false) {}
ConstraintsSolver::~ConstraintsSolver() {}

bool ConstraintsSolver::init(
    const nlohmann::json &config, std::vector<std::unique_ptr<PhysicsBaseObject>> &physics_objects
) {
    if (config.contains("epsilon")) epsilon = config["epsilon"].get<ScalarType>();
    if (config.contains("collision_radius")) collision_radius = config["collision_radius"].get<ScalarType>();
    if (config.contains("collision_grid_interval")) grid_interval = config["collision_grid_interval"].get<ScalarType>();
    if (config.contains("diagonal_scaling factor"))
        diagonal_scaling_factor = config["diagonal_scaling factor"].get<ScalarType>();
    if (config.contains("global_iter")) global_iter = config["global_iter"].get<unsigned int>();
    if (config.contains("normal_iter")) normal_iter = config["normal_iter"].get<unsigned int>();
    if (config.contains("tangential_iter")) tangential_iter = config["tangential_iter"].get<unsigned int>();
    if (config.contains("fixed_pos_iter")) fixed_pos_iter = config["fixed_pos_iter"].get<unsigned int>();
    if (config.contains("glue_iter")) glue_iter = config["glue_iter"].get<unsigned int>();
    if (config.contains("enable_collision")) enable_collision = config["enable_collision"].get<bool>();
    if (config.contains("enable_self_collision")) enable_self_collision = config["enable_self_collision"].get<bool>();
    if (config.contains("enable_glue")) enable_glue = config["enable_glue"].get<bool>();

    for (std::unique_ptr<PhysicsBaseObject> &physics_object : physics_objects) {
        physics_object_wrappers.emplace_back(*physics_object, grid_interval, enable_self_collision);

        if (physics_object->fixed_vertices.size() != 0) {
            position_constraints.emplace_back(physics_object_wrappers.size() - 1);
            enable_fixed_pos = true;
        }
    }

    init_glue_constraints();

    return true;
}

void ConstraintsSolver::init_glue_constraints() {
    if (!enable_glue) return;
    for (Eigen::Index pow1_idx = 0; pow1_idx < physics_object_wrappers.size(); pow1_idx++) {
        const auto &pow1 = physics_object_wrappers[pow1_idx];
        const auto &V1 = pow1.physics_object.get_V_estimate_global();
        const auto &F1 = pow1.physics_object.get_F();
        MatrixX3s FN1;
        igl::per_face_normals(V1, F1, FN1);
        for (Eigen::Index pow2_idx = 0; pow2_idx < pow1_idx; pow2_idx++) {
            const auto &pow2 = physics_object_wrappers[pow2_idx];
            const auto &V2 = pow2.physics_object.get_V_estimate_global();
            const auto &F2 = pow2.physics_object.get_F();
            MatrixX3s FN2;
            igl::per_face_normals(V2, F2, FN2);

            std::set<std::pair<Eigen::Index, Eigen::Index>> glue_vert_pairs;

            for (int f1 = 0; f1 < F1.rows(); f1++) {
                for (int f2 = 0; f2 < F2.rows(); f2++) {
                    if (FN1.row(f1).dot(FN2.row(f2)) > -0.9) continue;
                    for (int v1 = 0; v1 < 3; v1++) {
                        for (int v2 = 0; v2 < 3; v2++) {
                            if ((V1.row(F1(f1, v1)) - V2.row(F2(f2, v2))).norm() < epsilon)
                                glue_vert_pairs.insert({F1(f1, v1), F2(f2, v2)});
                        }
                    }
                }
            }

            std::vector<Eigen::Index> vertices1;
            std::vector<Eigen::Index> vertices2;
            for (auto [v1, v2] : glue_vert_pairs) {
                vertices1.emplace_back(v1);
                vertices2.emplace_back(v2);
            }

            if (vertices1.size() != 0) glue_constraints.emplace_back(pow1_idx, pow2_idx, vertices1, vertices2);
        }
    }
}

bool ConstraintsSolver::update_meshes() {
#pragma omp parallel for
    for (PhysicsObjectWrapper &pow : physics_object_wrappers) {
        if (pow.physics_object.is_fixed && pow.physics_object.is_static) continue;
        pow.aabb = Geometry::AABB(pow.physics_object.get_V_estimate_global());
        pow.update_triangle_spatial_map(grid_interval);
    }

    return true;
}

bool ConstraintsSolver::detect_collision() {
    if (!enable_collision) return true;

    // Make a map of (pow1, vertex1): all (pow, face) that collide with (pow1, vertex1)
    std::map<std::pair<size_t, Eigen::Index>, std::vector<std::pair<size_t, Eigen::Index>>> collision_vertex_faces;
    for (size_t pow1_idx = 0; pow1_idx < physics_object_wrappers.size(); pow1_idx++) {
        PhysicsObjectWrapper &pow1 = physics_object_wrappers[pow1_idx];
        const MatrixX3s &V1 = pow1.physics_object.get_V_estimate_global();
        for (size_t pow2_idx = 0; pow2_idx < physics_object_wrappers.size(); pow2_idx++) {
            PhysicsObjectWrapper &pow2 = physics_object_wrappers[pow2_idx];
            if (pow1_idx == pow2_idx && (!enable_self_collision || !pow1.physics_object.is_deformable)) continue;
            if (!pow1.aabb.intersects(pow2.aabb)) continue;

            const MatrixX3s &V2 = pow2.physics_object.get_V_estimate_global();
            const MatrixX3i &F2 = pow2.physics_object.get_F();
            auto &triangle_spatial_map = pow2.triangle_spatial_map;

#pragma omp parallel for
            for (Eigen::Index v = 0; v < V1.rows(); v++) {
                Vector3s vert1 = V1.row(v);
                Geometry::AABB aabb = Geometry::AABB(vert1, collision_radius);
                Vector3i min_coords = Geometry::point_to_grid(aabb.p_min, grid_interval);
                Vector3i max_coords = Geometry::point_to_grid(aabb.p_max, grid_interval);

                std::unordered_set<std::pair<size_t, Eigen::Index>, PAIR_HASH<std::pair<size_t, Eigen::Index>>>
                    collision_faces_set;

                for (auto x = min_coords[0]; x <= max_coords[0]; x++) {
                    for (auto y = min_coords[1]; y <= max_coords[1]; y++) {
                        for (auto z = min_coords[2]; z <= max_coords[2]; z++) {
                            Vector3i grid_coord(x, y, z);
                            if (triangle_spatial_map.count(grid_coord) == 0) continue;
                            const auto &faces = triangle_spatial_map[grid_coord];
                            for (auto f : faces) {
                                // If the face is already in the set, skip.
                                if (collision_faces_set.count({pow2_idx, f})) continue;

                                // If it is checking for self collision and the vertex is in the triangle, skip.
                                if (pow1_idx == pow2_idx && !(pow1.self_collision_matrix(v, f))) continue;

                                // If the sphere and the face intersects, add to map
                                if (Geometry::triangle_sphere_intersection(
                                        V2.row(F2(f, 0)), V2.row(F2(f, 1)), V2.row(F2(f, 2)), vert1, collision_radius
                                    ))
                                    collision_faces_set.insert({pow2_idx, f});
                            }
                        }
                    }
                }
                if (!collision_faces_set.empty()) {
#pragma omp critical
                    {
                        collision_vertex_faces.try_emplace({pow1_idx, v});
                        collision_vertex_faces[{pow1_idx, v}].insert(
                            collision_vertex_faces[{pow1_idx, v}].end(), collision_faces_set.begin(),
                            collision_faces_set.end()
                        );
                    }
                }
            }
        }
    }

    // Compute vertex normals
    {
        std::unordered_set<Eigen::Index> collision_physics_object_wrappers;
        for (const auto &[pow_vertex, pow_faces] : collision_vertex_faces)
            collision_physics_object_wrappers.insert(pow_vertex.first);

        // covert from unordered_set to vector for OpenMP loop parallelization.
        std::vector<Eigen::Index> collision_physics_object_wrappers_vector(
            collision_physics_object_wrappers.begin(), collision_physics_object_wrappers.end()
        );

#pragma omp parallel for
        for (Eigen::Index pow_idx : collision_physics_object_wrappers_vector) {
            PhysicsObjectWrapper &pow = physics_object_wrappers[pow_idx];
            const MatrixX3s &V = pow.physics_object.get_V_estimate_global();
            const MatrixX3i &F = pow.physics_object.get_F();
            igl::per_vertex_normals(V, F, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, pow.VN);
        }
    }

    // Make a map of (pow1, vertex1): best (pow, face, barycentric_coord) that collides with (pow1, vertex1)
    collision_constraints.clear();
#pragma omp parallel
    {
        size_t count = 0;
#ifdef OMP_AVAILABLE
        int ithread = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
#else
        int ithread = 0;
        int nthreads = 1;
#endif
        for (auto itr = collision_vertex_faces.begin(); itr != collision_vertex_faces.end(); itr++, count++) {
            if (count % nthreads != ithread) continue;

            const std::pair<size_t, Eigen::Index> &pow_vertex = itr->first;
            std::vector<std::pair<size_t, Eigen::Index>> pow_faces = itr->second;

            size_t vertex_pow_idx = pow_vertex.first;
            Eigen::Index vertex_idx = pow_vertex.second;
            PhysicsObjectWrapper &vertex_pow = physics_object_wrappers[vertex_pow_idx];

            RowVector3s vertex = vertex_pow.physics_object.get_V_estimate_global().row(vertex_idx);
            RowVector3s vertex_prev = vertex_pow.physics_object.get_V_global().row(vertex_idx);
            RowVector3s VN = vertex_pow.VN.row(vertex_idx);

            std::vector<std::pair<ScalarType, std::tuple<Eigen::Index, Eigen::Index, RowVector3s>>> candidate_faces;
            for (auto &[face_pow_idx, face_idx] : pow_faces) {
                if (count % nthreads != ithread) continue;

                PhysicsObjectWrapper &face_pow = physics_object_wrappers[face_pow_idx];

                auto face = face_pow.physics_object.get_F().row(face_idx);
                const MatrixX3s &V = face_pow.physics_object.get_V_estimate_global();
                RowVector3s v1 = V.row(face(0)), v2 = V.row(face(1)), v3 = V.row(face(2));
                RowVector3s n = ((v2 - v1).cross(v3 - v1)).normalized();
                if (n.dot(VN) > -0.5) continue;

                // RowVector3s contact_point = vertex - signed_dist * n;
                RowVector3s contact_point = Geometry::closest_line_plane(vertex_prev, vertex, v1, n);
                RowVector3s bc;
                igl::barycentric_coordinates(contact_point, v1, v2, v3, bc);
                if (!((bc.array() > -1e-3).all() && (bc.array() < 1. + 1e-3).all())) continue;

                const MatrixX3s &face_prev_V = face_pow.physics_object.get_V_global();
                Vector3s v_rel_motion = vertex - vertex_prev -
                                        ((v1 * bc(0) + v2 * bc(1) + v3 * bc(2)) -
                                         (face_prev_V.row(face(0)) * bc(0) + face_prev_V.row(face(1)) * bc(1) +
                                          face_prev_V.row(face(2)) * bc(2)));
                if (n.dot(v_rel_motion) >= 0.0) continue;

                ScalarType signed_dist = (vertex - v1).dot(n);
                // if (signed_dist >= 0.0) continue;

                ScalarType score = signed_dist;
                candidate_faces.push_back(
                    {score, std::tuple<Eigen::Index, Eigen::Index, RowVector3s>(face_pow_idx, face_idx, bc)}
                );
            }
            if (!candidate_faces.size()) continue;

            {
                std::sort(candidate_faces.begin(), candidate_faces.end(), [](const auto &pair1, const auto &pair2) {
                    return pair1.first < pair2.first;
                }); // sort by score (ascending order)

                ScalarType min_score = candidate_faces[0].first;
                auto itr = candidate_faces.begin();
                for (; itr != candidate_faces.end(); itr++) {
                    if (itr->first > min_score + 1e-3) { break; }
                }
                candidate_faces.erase(itr, candidate_faces.end());
            }

#pragma omp critical
            {
                for (auto &[score, face_data] : candidate_faces) {
                    const auto &[face_pow_idx, face_idx, bc] = face_data;
                    collision_constraints.try_emplace({vertex_pow_idx, face_pow_idx});
                    collision_constraints.try_emplace({face_pow_idx, vertex_pow_idx});
                    collision_constraints[{vertex_pow_idx, face_pow_idx}].emplace_back(
                        vertex_idx, std::make_pair(face_idx, bc)
                    );
                }
            }
        }
    }
    return true;
}

void ConstraintsSolver::precompute() {

    constraints_data_vector.clear();

    // compute data for each type of constraint
#pragma omp parallel for
    for (int i = 0; i < 3; i++) {
        if (i == 0 && enable_collision) precompute_collision_constraints();
        if (i == 1 && enable_glue) precompute_glue_constraints();
        if (i == 2 && enable_fixed_pos) precompute_fixed_pos_constraints();
    }

    // make batches for constraints solution
    constraints_solution_batches.clear();
    {
        VectorXi not_processed = VectorXi::Ones(constraints_data_vector.size());
        // while not every constraint is included in a batch
        while (not_processed.sum() > 0) {
            std::vector<std::reference_wrapper<ConstraintData>> &batch = constraints_solution_batches.emplace_back();
            for (size_t i = 0; i < constraints_data_vector.size(); i++) {
                ConstraintData &data = constraints_data_vector[i];
                if (not_processed(i)) {
                    bool valid = true;

                    bool data_object1_can_be_invalid =
                        !physics_object_wrappers[data.object1_idx].physics_object.is_fixed ||
                        physics_object_wrappers[data.object1_idx].physics_object.is_deformable;
                    bool data_object2_can_be_invalid =
                        !data.is_single_object_contraint &&
                        (!physics_object_wrappers[data.object2_idx].physics_object.is_fixed ||
                         physics_object_wrappers[data.object2_idx].physics_object.is_deformable);

                    for (ConstraintData &batch_entry : batch) {
                        bool batch_entry_object1_can_be_invalid =
                            !physics_object_wrappers[batch_entry.object1_idx].physics_object.is_fixed ||
                            physics_object_wrappers[batch_entry.object1_idx].physics_object.is_deformable;
                        bool batch_entry_object2_can_be_invalid =
                            !batch_entry.is_single_object_contraint &&
                            (!physics_object_wrappers[batch_entry.object2_idx].physics_object.is_fixed ||
                             physics_object_wrappers[batch_entry.object2_idx].physics_object.is_deformable);

                        if ((data_object1_can_be_invalid && batch_entry_object1_can_be_invalid &&
                             data.object1_idx == batch_entry.object1_idx) ||
                            (data_object2_can_be_invalid && batch_entry_object1_can_be_invalid &&
                             data.object2_idx == batch_entry.object1_idx) ||
                            (data_object1_can_be_invalid && batch_entry_object2_can_be_invalid &&
                             data.object1_idx == batch_entry.object2_idx) ||
                            (data_object2_can_be_invalid && batch_entry_object2_can_be_invalid &&
                             data.object2_idx == batch_entry.object2_idx)) {
                            valid = false;
                            break;
                        }
                    }

                    if (valid) {
                        not_processed[i] = 0;
                        batch.emplace_back(data);
                    }
                }
                // i++;
            }
        }
    }
}

void ConstraintsSolver::precompute_collision_constraints() {
    auto get_col_indices = [](const SparseMatrix &mat) {
        std::unordered_set<int> col_set;
        for (int i = 0; i < mat.nonZeros(); i++) col_set.emplace(mat.innerIndexPtr()[i]);
        VectorXi cols(col_set.size());
        int i = 0;
        for (int col : col_set) { cols(i++) = col; }
        return cols;
    };

    std::vector<std::pair<size_t, size_t>> collision_constraints_key_list;
    collision_constraints_key_list.reserve(collision_constraints.size());
    for (const auto &[pow_pair, data1] : collision_constraints) {
        collision_constraints_key_list.emplace_back(pow_pair);
        const size_t pow1_idx = pow_pair.first;
        const size_t pow2_idx = pow_pair.second;

        PhysicsBaseObject &po1 = physics_object_wrappers[pow1_idx].physics_object;
        PhysicsBaseObject &po2 = physics_object_wrappers[pow2_idx].physics_object;

        if (!po1.has_constraints) {
            po1.update_V_estimate_matrices();
            po1.has_constraints = true;
        }
        if (!po2.has_constraints) {
            po2.update_V_estimate_matrices();
            po2.has_constraints = true;
        }
    }

#pragma omp parallel for
    for (size_t i = 0; i < collision_constraints_key_list.size(); i++) {
        const auto &pow_pair = collision_constraints_key_list[i];
        const size_t pow1_idx = pow_pair.first;
        const size_t pow2_idx = pow_pair.second;
        if (pow1_idx > pow2_idx) continue;

        bool is_self_collision = pow1_idx == pow2_idx;

        const auto &data1 = collision_constraints[pow_pair];
        const auto &data2 = collision_constraints[{pow2_idx, pow1_idx}];

        PhysicsObjectWrapper &pow1 = physics_object_wrappers[pow1_idx];
        PhysicsObjectWrapper &pow2 = physics_object_wrappers[pow2_idx];
        PhysicsBaseObject &po1 = pow1.physics_object;
        PhysicsBaseObject &po2 = pow2.physics_object;

        auto collision_vertex_indices = [](const auto &data) {
            std::vector<Eigen::Index> res;
            res.reserve(data.size());
            for (const auto &[vertex_idx, _] : data) res.emplace_back(vertex_idx);
            return res;
        };
        auto c1_vertex_indices = collision_vertex_indices(data1);
        auto c2_vertex_indices = is_self_collision ? std::vector<Eigen::Index>() : collision_vertex_indices(data2);

        auto collision_face_indices = [](const auto &data) {
            std::vector<Eigen::Index> res;
            res.reserve(data.size());
            for (const auto &[_, _data] : data) res.emplace_back(_data.first);
            return res;
        };
        auto c1_face_indices = collision_face_indices(data1);
        auto c2_face_indices = is_self_collision ? std::vector<Eigen::Index>() : collision_face_indices(data2);

        auto collision_normal_vectors = [&](const auto &data, size_t F_pow_index) {
            MatrixX3s N(data.size(), 3);
            size_t i = 0;
            for (const auto &[_, _data] : data) {
                RowVector3s n;
                igl::per_face_normals(
                    physics_object_wrappers[F_pow_index].physics_object.get_V_estimate_global(),
                    physics_object_wrappers[F_pow_index].physics_object.get_F().row(_data.first), n
                );
                N.row(i) = n;
                i++;
            }
            return N;
        };
        MatrixX3s c1_normal_vectors = collision_normal_vectors(data1, pow2_idx);
        MatrixX3s c2_normal_vectors = is_self_collision ? MatrixX3s(0, 3) : collision_normal_vectors(data2, pow1_idx);

        auto collision_barycentric_coords = [](const auto &data) {
            MatrixX3s BC(data.size(), 3);
            size_t i = 0;
            for (const auto &[_, _data] : data) {
                BC.row(i) = _data.second;
                i++;
            }
            return BC;
        };
        MatrixX3s c1_barycentric_coords = collision_barycentric_coords(data1);
        MatrixX3s c2_barycentric_coords = is_self_collision ? MatrixX3s(0, 3) : collision_barycentric_coords(data2);

        SparseMatrix f1_f1c, f1_f2c, f1c_f1N, f1c_f1F, N1_V1, N2_V1, F1_V1, F2_V1, f2_f2c, f2_f1c, f2c_f2N, f2c_f2F,
            N2_V2, N1_V2, F2_V2, F1_V2;
        po1.update_conversion_matrices(
            c1_vertex_indices, c1_normal_vectors, c2_face_indices, c2_normal_vectors, c2_barycentric_coords,
            po2.get_rotation(), f1_f1c, f1_f2c, f1c_f1N, f1c_f1F, N1_V1, N2_V1, F1_V1, F2_V1
        );
        po2.update_conversion_matrices(
            c2_vertex_indices, c2_normal_vectors, c1_face_indices, c1_normal_vectors, c1_barycentric_coords,
            po1.get_rotation(), f2_f2c, f2_f1c, f2c_f2N, f2c_f2F, N2_V2, N1_V2, F2_V2, F1_V2
        );

        const DiagonalMatrixXs &p1_f1 = po1.p_f, &p2_f2 = po2.p_f;
        const VectorXs &V1_prev = Eigen::Map<const VectorXs>(po1.get_V_global().data(), po1.get_V_global().size()),
                       &V2_prev = Eigen::Map<const VectorXs>(po2.get_V_global().data(), po2.get_V_global().size()),
                       &V1_const = po1.V_const, &V2_const = po2.V_const;

        FrictionCombine friction_combine = std::max(po1.friction_combine, po2.friction_combine);
        ScalarType friction_coeff;
        switch (friction_combine) {
        case Average: friction_coeff = (po1.friction_coeff + po2.friction_coeff) / 2.0; break;
        case Minimum: friction_coeff = std::min(po1.friction_coeff, po2.friction_coeff); break;
        case Multiply: friction_coeff = po1.friction_coeff * po2.friction_coeff; break;
        case Maximum: friction_coeff = std::max(po1.friction_coeff, po2.friction_coeff); break;
        }

        SparseMatrix N_V1(N1_V1.rows() + N2_V1.rows(), N1_V1.cols());
        N_V1.topRows(N1_V1.rows()) = N1_V1;
        N_V1.bottomRows(N2_V1.rows()) = -N2_V1;

        SparseMatrix N_V2(N1_V2.rows() + N2_V2.rows(), N1_V2.cols());
        N_V2.topRows(N1_V2.rows()) = N1_V2;
        N_V2.bottomRows(N2_V2.rows()) = -N2_V2;

        SparseMatrix F_V1(F1_V1.rows() + F2_V1.rows(), F1_V1.cols());
        F_V1.topRows(F1_V1.rows()) = F1_V1;
        F_V1.bottomRows(F2_V1.rows()) = -F2_V1;

        SparseMatrix F_V2(F1_V2.rows() + F2_V2.rows(), F1_V2.cols());
        F_V2.topRows(F1_V2.rows()) = F1_V2;
        F_V2.bottomRows(F2_V2.rows()) = -F2_V2;

        size_t num_collisions = f1c_f1N.cols() + f2c_f2N.cols();

        Eigen::SparseMatrix<ScalarType, Eigen::ColMajor> f1_fN(p1_f1.rows(), num_collisions);
        f1_fN.leftCols(f1c_f1N.cols()) = f1_f1c * f1c_f1N;
        f1_fN.rightCols(f2c_f2N.cols()) = f1_f2c * f2c_f2N;

        Eigen::SparseMatrix<ScalarType, Eigen::ColMajor> f2_fN(p2_f2.rows(), num_collisions);
        f2_fN.leftCols(f1c_f1N.cols()) = f2_f1c * f1c_f1N;
        f2_fN.rightCols(f2c_f2N.cols()) = f2_f2c * f2c_f2N;

        Eigen::SparseMatrix<ScalarType, Eigen::ColMajor> f1_fF(p1_f1.rows(), num_collisions * 2);
        f1_fF.leftCols(f1c_f1F.cols()) = f1_f1c * f1c_f1F;
        f1_fF.rightCols(f2c_f2F.cols()) = f1_f2c * f2c_f2F;

        Eigen::SparseMatrix<ScalarType, Eigen::ColMajor> f2_fF(p2_f2.rows(), num_collisions * 2);
        f2_fF.leftCols(f1c_f1F.cols()) = f2_f1c * f1c_f1F;
        f2_fF.rightCols(f2c_f2F.cols()) = f2_f2c * f2c_f2F;

        SparseMatrix p1_fN = p1_f1 * f1_fN, p2_fN = p2_f2 * f2_fN, p1_fF = p1_f1 * f1_fF, p2_fF = p2_f2 * f2_fF;

        const std::shared_ptr<const PhysicsBaseMatrixCollection> &matrix_collection1 = po1.matrix_collection;
        const std::shared_ptr<const PhysicsBaseMatrixCollection> &matrix_collection2 = po2.matrix_collection;

        MatrixXs N_fN = (N_V1 * po1.rot_trans_acc) * (matrix_collection1->translational_acc_mat * p1_fN) +
                        (N_V1 * po1.rot_rot_acc) * (matrix_collection1->rotational_acc_mat * p1_fN) -
                        (N_V2 * po2.rot_trans_acc) * (matrix_collection2->translational_acc_mat * p2_fN) -
                        (N_V2 * po2.rot_rot_acc) * (matrix_collection2->rotational_acc_mat * p2_fN);

        MatrixXs F_fF = (F_V1 * po1.rot_trans_acc) * (matrix_collection1->translational_acc_mat * p1_fF) +
                        (F_V1 * po1.rot_rot_acc) * (matrix_collection1->rotational_acc_mat * p1_fF) -
                        (F_V2 * po2.rot_trans_acc) * (matrix_collection2->translational_acc_mat * p2_fF) -
                        (F_V2 * po2.rot_rot_acc) * (matrix_collection2->rotational_acc_mat * p2_fF);

        MatrixXs N_p1 = (N_V1 * po1.rot_trans_acc) * matrix_collection1->translational_acc_mat +
                        (N_V1 * po1.rot_rot_acc) * matrix_collection1->rotational_acc_mat;

        MatrixXs N_p2 = (N_V2 * po2.rot_trans_acc) * matrix_collection2->translational_acc_mat +
                        (N_V2 * po2.rot_rot_acc) * matrix_collection2->rotational_acc_mat;

        MatrixXs F_p1 = (F_V1 * po1.rot_trans_acc) * matrix_collection1->translational_acc_mat +
                        (F_V1 * po1.rot_rot_acc) * matrix_collection1->rotational_acc_mat;

        MatrixXs F_p2 = (F_V2 * po2.rot_trans_acc) * matrix_collection2->translational_acc_mat +
                        (F_V2 * po2.rot_rot_acc) * matrix_collection2->rotational_acc_mat;

        VectorXs A_NN_diag = (N_V1 * po1.V_u * matrix_collection1->u_p_diag * p1_fN).eval().diagonal() -
                             (N_V2 * po2.V_u * matrix_collection2->u_p_diag * p2_fN).eval().diagonal() +
                             N_fN.diagonal();

        VectorXs A_FF_diag = (F_V1 * po1.V_u * matrix_collection1->u_p_diag * p1_fF).eval().diagonal() -
                             (F_V2 * po2.V_u * matrix_collection2->u_p_diag * p2_fF).eval().diagonal() +
                             F_fF.diagonal();

        SparseMatrix N_u1_comp = N_V1 * po1.V_u * matrix_collection1->compression_permutation.transpose();
        SparseMatrix N_u2_comp = N_V2 * po2.V_u * matrix_collection2->compression_permutation.transpose();
        SparseMatrix F_u1_comp = F_V1 * po1.V_u * matrix_collection1->compression_permutation.transpose();
        SparseMatrix F_u2_comp = F_V2 * po2.V_u * matrix_collection2->compression_permutation.transpose();

        CollisionConstraintData *collision_constraint_data_ptr = new CollisionConstraintData{
            friction_coeff,

            po1.is_deformable,
            po2.is_deformable,
            matrix_collection1->use_compressed_matrices,
            matrix_collection2->use_compressed_matrices,

            f1_fN,
            f2_fN,
            f1_fF,
            f2_fF,
            p1_fN,
            p2_fN,
            p1_fF,
            p2_fF,

            A_NN_diag,
            A_FF_diag,
            N_fN,
            F_fF,
            N_p1,
            N_p2,
            F_p1,
            F_p2,

            N_u1_comp,
            matrix_collection1->compression_permutation * p1_fN,
            N_u2_comp,
            matrix_collection2->compression_permutation * p2_fN,
            F_u1_comp,
            matrix_collection1->compression_permutation * p1_fF,
            F_u2_comp,
            matrix_collection2->compression_permutation * p2_fF,

            (matrix_collection1->use_compressed_matrices || matrix_collection2->use_compressed_matrices)
                ? get_col_indices(N_u1_comp)
                : VectorXi(),
            (matrix_collection1->use_compressed_matrices || matrix_collection2->use_compressed_matrices)
                ? get_col_indices(N_u2_comp)
                : VectorXi(),
            (matrix_collection1->use_compressed_matrices || matrix_collection2->use_compressed_matrices)
                ? get_col_indices(F_u1_comp)
                : VectorXi(),
            (matrix_collection1->use_compressed_matrices || matrix_collection2->use_compressed_matrices)
                ? get_col_indices(F_u2_comp)
                : VectorXi(),

            p1_f1,
            matrix_collection1->u_p_compressed,
            matrix_collection1->u_p,
            matrix_collection1->compression_permutation,
            p2_f2,
            matrix_collection2->u_p_compressed,
            matrix_collection2->u_p,
            matrix_collection2->compression_permutation,

            /*b_N*/ N_V1 * V1_const - N_V2 * V2_const,
            /*b_F*/ F_V1 * (V1_const - V1_prev) - F_V2 * (V2_const - V2_prev),

            /*fN*/ VectorXs::Zero(num_collisions),
            /*fF*/ VectorXs::Zero(num_collisions * 2)};

#pragma omp critical
        {
            constraints_data_vector.push_back({COLLISION, false, pow1_idx, pow2_idx});
            constraints_data_vector.back().collision_constraint_data.reset(collision_constraint_data_ptr);
            pow1.f.setZero();
            pow2.f.setZero();
        }
    }
}

void ConstraintsSolver::precompute_glue_constraints() {
    auto get_col_indices = [](const SparseMatrix &mat) {
        std::unordered_set<int> col_set;
        for (int i = 0; i < mat.nonZeros(); i++) col_set.emplace(mat.innerIndexPtr()[i]);
        VectorXi cols(col_set.size());
        int i = 0;
        for (int col : col_set) { cols(i++) = col; }
        return cols;
    };

    for (const auto &[pow1_idx, pow2_idx, _1, _2] : glue_constraints) {
        PhysicsBaseObject &po1 = physics_object_wrappers[pow1_idx].physics_object;
        PhysicsBaseObject &po2 = physics_object_wrappers[pow2_idx].physics_object;

        if (!po1.has_constraints) {
            po1.update_V_estimate_matrices();
            po1.has_constraints = true;
        }
        if (!po2.has_constraints) {
            po2.update_V_estimate_matrices();
            po2.has_constraints = true;
        }
    }

#pragma omp parallel for
    for (const auto &[_pow1_idx, _pow2_idx, vertices1, vertices2] : glue_constraints) {
        size_t pow1_idx = _pow1_idx;
        PhysicsObjectWrapper &pow1 = physics_object_wrappers[pow1_idx];
        PhysicsBaseObject &po1 = pow1.physics_object;
        const std::shared_ptr<const PhysicsBaseMatrixCollection> &matrix_collection1 = po1.matrix_collection;

        Eigen::Index num_vertices1 = po1.get_num_vertices();

        SparseMatrix P_V1(vertices1.size() * 3, num_vertices1 * 3);
        {
            std::vector<Eigen::Triplet<ScalarType>> elems;
            elems.reserve(vertices1.size() * 3);
            Eigen::Index v_new = 0;
            for (auto v : vertices1) {
                elems.emplace_back(3 * v_new, 3 * v, 1.0);
                elems.emplace_back(3 * v_new + 1, 3 * v + 1, 1.0);
                elems.emplace_back(3 * v_new + 2, 3 * v + 2, 1.0);
                v_new++;
            }
            P_V1.setFromTriplets(elems.begin(), elems.end());
        }

        SparseMatrix f1_fP(num_vertices1 * 3, vertices1.size() * 3);
        {
            std::vector<Eigen::Triplet<ScalarType>> elems;
            elems.reserve(vertices1.size() * 3);
            Eigen::Index v_new = 0;
            for (auto v : vertices1) {
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) { elems.emplace_back(3 * v + i, 3 * v_new + j, po1.rotation(j, i)); }
                }
                v_new++;
            }
            f1_fP.setFromTriplets(elems.begin(), elems.end());
        }

        SparseMatrix P_u1_comp = P_V1 * po1.V_u * matrix_collection1->compression_permutation.transpose();
        SparseMatrix p1_comp_fP = matrix_collection1->compression_permutation * (SparseMatrix)(po1.p_f * f1_fP);

        MatrixXs P_p1 = (P_V1 * po1.rot_trans_acc) * (matrix_collection1->translational_acc_mat) +
                        (P_V1 * po1.rot_rot_acc) * (matrix_collection1->rotational_acc_mat);

        size_t pow2_idx = _pow2_idx;
        PhysicsObjectWrapper &pow2 = physics_object_wrappers[pow2_idx];
        PhysicsBaseObject &po2 = pow2.physics_object;
        const std::shared_ptr<const PhysicsBaseMatrixCollection> &matrix_collection2 = po2.matrix_collection;

        Eigen::Index num_vertices2 = po2.get_num_vertices();

        SparseMatrix P_V2(vertices2.size() * 3, num_vertices2 * 3);
        {
            std::vector<Eigen::Triplet<ScalarType>> elems;
            elems.reserve(vertices2.size() * 3);
            Eigen::Index v_new = 0;
            for (auto v : vertices2) {
                elems.emplace_back(3 * v_new, 3 * v, 1.0);
                elems.emplace_back(3 * v_new + 1, 3 * v + 1, 1.0);
                elems.emplace_back(3 * v_new + 2, 3 * v + 2, 1.0);
                v_new++;
            }
            P_V2.setFromTriplets(elems.begin(), elems.end());
        }

        SparseMatrix f2_fP(num_vertices2 * 3, vertices2.size() * 3);
        {
            std::vector<Eigen::Triplet<ScalarType>> elems;
            elems.reserve(vertices2.size() * 3);
            Eigen::Index v_new = 0;
            for (auto v : vertices2) {
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) { elems.emplace_back(3 * v + i, 3 * v_new + j, -po2.rotation(j, i)); }
                }
                v_new++;
            }
            f2_fP.setFromTriplets(elems.begin(), elems.end());
        }

        SparseMatrix P_u2_comp = P_V2 * po2.V_u * matrix_collection2->compression_permutation.transpose();
        SparseMatrix p2_comp_fP = matrix_collection2->compression_permutation * (SparseMatrix)(po2.p_f * f2_fP);

        MatrixXs P_p2 = (P_V2 * po2.rot_trans_acc) * (matrix_collection2->translational_acc_mat) +
                        (P_V2 * po2.rot_rot_acc) * (matrix_collection2->rotational_acc_mat);

        MatrixXs P_fP = (P_V1 * po1.rot_trans_acc) * (matrix_collection1->translational_acc_mat * po1.p_f * f1_fP) +
                        (P_V1 * po1.rot_rot_acc) * (matrix_collection1->rotational_acc_mat * po1.p_f * f1_fP) -
                        (P_V2 * po2.rot_trans_acc) * (matrix_collection2->translational_acc_mat * po2.p_f * f2_fP) -
                        (P_V2 * po2.rot_rot_acc) * (matrix_collection2->rotational_acc_mat * po2.p_f * f2_fP);

        VectorXs A_PP_diag = (P_V1 * po1.V_u * matrix_collection1->u_p_diag * po1.p_f * f1_fP).eval().diagonal() -
                             (P_V2 * po2.V_u * matrix_collection2->u_p_diag * po2.p_f * f2_fP).eval().diagonal() +
                             P_fP.diagonal();

        GlueConstraintData *glue_constraint_data_ptr = new GlueConstraintData{
            po1.is_deformable,
            po2.is_deformable,
            matrix_collection1->use_compressed_matrices,
            matrix_collection2->use_compressed_matrices,
            f1_fP,
            f2_fP,
            A_PP_diag,
            P_fP,
            P_p1,
            P_p2,
            P_u1_comp,
            p1_comp_fP,
            P_u2_comp,
            p2_comp_fP,
            (matrix_collection1->use_compressed_matrices || matrix_collection2->use_compressed_matrices)
                ? get_col_indices(P_u1_comp)
                : VectorXi(),
            (matrix_collection1->use_compressed_matrices || matrix_collection2->use_compressed_matrices)
                ? get_col_indices(P_u2_comp)
                : VectorXi(),

            po1.p_f,
            po2.p_f,
            matrix_collection1->u_p_compressed,
            matrix_collection2->u_p_compressed,
            matrix_collection1->u_p,
            matrix_collection2->u_p,
            matrix_collection1->compression_permutation,
            matrix_collection2->compression_permutation,
            /*b*/ P_V1 * po1.V_const - P_V2 * po2.V_const,

            /*fP*/ VectorXs::Zero(vertices1.size() * 3),
        };

#pragma omp critical
        {
            constraints_data_vector.push_back({GLUE, false, pow1_idx, pow2_idx});
            constraints_data_vector.back().glue_constraint_data.reset(glue_constraint_data_ptr);
            pow1.f.setZero();
            pow2.f.setZero();
        }
    }
}

void ConstraintsSolver::precompute_fixed_pos_constraints() {
    auto get_col_indices = [](const SparseMatrix &mat) {
        std::unordered_set<int> col_set;
        for (int i = 0; i < mat.nonZeros(); i++) col_set.emplace(mat.innerIndexPtr()[i]);
        VectorXi cols(col_set.size());
        int i = 0;
        for (int col : col_set) { cols(i++) = col; }
        return cols;
    };

    for (const size_t pow_idx : position_constraints) {
        PhysicsBaseObject &po = physics_object_wrappers[pow_idx].physics_object;
        if (!po.has_constraints) {
            po.update_V_estimate_matrices();
            po.has_constraints = true;
        }
    }

#pragma omp parallel for
    for (const size_t pow_idx : position_constraints) {
        // size_t pow_idx = _pow_idx;
        PhysicsObjectWrapper &pow = physics_object_wrappers[pow_idx];
        PhysicsBaseObject &po = pow.physics_object;
        const std::vector<Eigen::Index> &vertices = po.fixed_vertices;

        Eigen::Index num_vertices = pow.physics_object.get_num_vertices();

        SparseMatrix P_V(vertices.size() * 3, num_vertices * 3);
        {
            std::vector<Eigen::Triplet<ScalarType>> elems;
            elems.reserve(vertices.size() * 3);
            Eigen::Index v_new = 0;
            for (auto v : vertices) {
                elems.emplace_back(3 * v_new, 3 * v, 1.0);
                elems.emplace_back(3 * v_new + 1, 3 * v + 1, 1.0);
                elems.emplace_back(3 * v_new + 2, 3 * v + 2, 1.0);
                v_new++;
            }
            P_V.setFromTriplets(elems.begin(), elems.end());
        }

        SparseMatrix f_fP(num_vertices * 3, vertices.size() * 3);
        {
            std::vector<Eigen::Triplet<ScalarType>> elems;
            elems.reserve(vertices.size() * 3);
            Eigen::Index v_new = 0;
            for (auto v : vertices) {
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) { elems.emplace_back(3 * v + i, 3 * v_new + j, po.rotation(j, i)); }
                }
                v_new++;
            }
            f_fP.setFromTriplets(elems.begin(), elems.end());
        }

        const std::shared_ptr<const PhysicsBaseMatrixCollection> &matrix_collection = po.matrix_collection;

        SparseMatrix P_u_comp = P_V * po.V_u * matrix_collection->compression_permutation.transpose();
        SparseMatrix p_comp_fP = matrix_collection->compression_permutation * (SparseMatrix)(po.p_f * f_fP);

        MatrixXs P_fP = (P_V * po.rot_trans_acc) * (matrix_collection->translational_acc_mat * po.p_f * f_fP) +
                        (P_V * po.rot_rot_acc) * (matrix_collection->rotational_acc_mat * po.p_f * f_fP);
        MatrixXs P_p = (P_V * po.rot_trans_acc) * (matrix_collection->translational_acc_mat) +
                       (P_V * po.rot_rot_acc) * (matrix_collection->rotational_acc_mat);

        VectorXs A_PP_diag =
            (P_V * po.V_u * matrix_collection->u_p_diag * po.p_f * f_fP).eval().diagonal() + P_fP.diagonal();

        FixedPosConstraintData *fixed_pos_constraint_data_ptr = new FixedPosConstraintData{
            po.is_deformable,
            matrix_collection->use_compressed_matrices,
            f_fP,
            A_PP_diag,
            P_fP,
            P_p,
            P_u_comp,
            p_comp_fP,
            matrix_collection->use_compressed_matrices ? get_col_indices(P_u_comp) : VectorXi(),

            po.p_f,
            matrix_collection->u_p_compressed,
            matrix_collection->u_p,
            matrix_collection->compression_permutation,

            /*b*/ P_V * (po.V_const - pow.V_init),

            /*fP*/ VectorXs::Zero(vertices.size() * 3),
        };

#pragma omp critical
        {
            constraints_data_vector.push_back({FIXED_POS, true, pow_idx, -1});
            constraints_data_vector.back().fixed_pos_constraint_data.reset(fixed_pos_constraint_data_ptr);
            pow.f.setZero();
        }
    }
}

bool ConstraintsSolver::solve_collision_constraints(ConstraintData &constraint_data) {
    auto get_col_indices = [](const SparseMatrix &mat, int row) {
        auto size = mat.outerIndexPtr()[row + 1] - mat.outerIndexPtr()[row];
        auto ptr = mat.innerIndexPtr() + mat.outerIndexPtr()[row];
        return Eigen::Map<const VectorXi>(ptr, size);
    };

    bool updated = false;

    CollisionConstraintData &data = *constraint_data.collision_constraint_data;
    VectorXs &f1 = physics_object_wrappers[constraint_data.object1_idx].f;
    VectorXs &f2 = physics_object_wrappers[constraint_data.object2_idx].f;
    VectorXs &fN = data.fN;
    VectorXs &fF = data.fF;

    size_t num_collisions = fN.rows();

    f1 -= data.f1_fN * fN + data.f1_fF * fF;
    f2 -= data.f2_fN * fN + data.f2_fF * fF;

    {
        // tangential direction
        if (data.friction_coeff > 0.0) {
            VectorXs p1 = data.p1_fN * fN + data.p1_f1 * f1, p2 = data.p2_fN * fN + data.p2_f2 * f2;
            VectorXs b = -data.F_p1 * p1 + data.F_p2 * p2 - data.b_F;
            if (data.is_deformable1) {
                if (data.use_compressed_matrix1)
                    b -= data.F_u1_comp * CompressedMatrixVectorProduct<>::rows(
                                              data.u1_p1, CompressedVector<>(data.comp_perm1 * p1), data.F_u1_cols
                                          );
                else
                    b -= data.F_u1_comp * data.u1_p1_uncompressed * p1;
            }
            if (data.is_deformable2) {
                if (data.use_compressed_matrix2)
                    b += data.F_u2_comp * CompressedMatrixVectorProduct<>::rows(
                                              data.u2_p2, CompressedVector<>(data.comp_perm2 * p2), data.F_u2_cols
                                          );
                else
                    b += data.F_u2_comp * data.u2_p2_uncompressed * p2;
            }

            CompressedVector<> p1_fF_comp;
            if (data.use_compressed_matrix1) p1_fF_comp = CompressedVector<>(data.p1_comp_fF * fF);
            CompressedVector<> p2_fF_comp;
            if (data.use_compressed_matrix2) p2_fF_comp = CompressedVector<>(data.p2_comp_fF * fF);

            for (unsigned int j = 0; j < tangential_iter; j++) {
                VectorXs fF_old = fF;

                for (int v = 0; v < num_collisions; v++) {
                    auto update_indices01 = data.use_compressed_matrix1 ? get_col_indices(data.F_u1_comp, 2 * v)
                                                                        : Eigen::Map<const VectorXi>(nullptr, 0);
                    auto update_indices02 = data.use_compressed_matrix2 ? get_col_indices(data.F_u2_comp, 2 * v)
                                                                        : Eigen::Map<const VectorXi>(nullptr, 0);

                    ScalarType b_v1 = b(2 * v) - data.F_fF.row(2 * v) * fF -
                                      diagonal_scaling_factor * data.A_FF_diag(2 * v) * fF(2 * v);
                    if (data.is_deformable1) {
                        if (data.use_compressed_matrix1)
                            b_v1 -= data.F_u1_comp.row(2 * v) *
                                    CompressedMatrixVectorProduct<>::rows(data.u1_p1, p1_fF_comp, update_indices01);
                        else
                            b_v1 -= data.F_u1_comp.row(2 * v) * data.u1_p1_uncompressed * data.p1_comp_fF * fF;
                    }
                    if (data.is_deformable2) {
                        if (data.use_compressed_matrix2)
                            b_v1 += data.F_u2_comp.row(2 * v) *
                                    CompressedMatrixVectorProduct<>::rows(data.u2_p2, p2_fF_comp, update_indices02);
                        else
                            b_v1 += data.F_u2_comp.row(2 * v) * data.u2_p2_uncompressed * data.p2_comp_fF * fF;
                    }

                    auto update_indices11 = data.use_compressed_matrix1 ? get_col_indices(data.F_u1_comp, 2 * v + 1)
                                                                        : Eigen::Map<const VectorXi>(nullptr, 0);
                    auto update_indices12 = data.use_compressed_matrix2 ? get_col_indices(data.F_u2_comp, 2 * v + 1)
                                                                        : Eigen::Map<const VectorXi>(nullptr, 0);
                    ScalarType b_v2 = b(2 * v + 1) - data.F_fF.row(2 * v + 1) * fF -
                                      diagonal_scaling_factor * data.A_FF_diag(2 * v + 1) * fF(2 * v + 1);
                    if (data.is_deformable1) {
                        if (data.use_compressed_matrix1)
                            b_v2 -= data.F_u1_comp.row(2 * v + 1) *
                                    CompressedMatrixVectorProduct<>::rows(data.u1_p1, p1_fF_comp, update_indices11);
                        else
                            b_v2 -= data.F_u1_comp.row(2 * v + 1) * data.u1_p1_uncompressed * data.p1_comp_fF * fF;
                    }
                    if (data.is_deformable2) {
                        if (data.use_compressed_matrix2)
                            b_v2 += data.F_u2_comp.row(2 * v + 1) *
                                    CompressedMatrixVectorProduct<>::rows(data.u2_p2, p2_fF_comp, update_indices12);
                        else
                            b_v2 += data.F_u2_comp.row(2 * v + 1) * data.u2_p2_uncompressed * data.p2_comp_fF * fF;
                    }

                    fF(2 * v) += b_v1 / (data.A_FF_diag(2 * v) * (1. + diagonal_scaling_factor));
                    fF(2 * v + 1) += b_v2 / (data.A_FF_diag(2 * v + 1) * (1. + diagonal_scaling_factor));

                    if (fF.segment<2>(2 * v).squaredNorm() >
                        (data.friction_coeff * fN(v)) * (data.friction_coeff * fN(v)))
                        fF.segment<2>(2 * v) = fF.segment<2>(2 * v).stableNormalized() * data.friction_coeff * fN(v);

                    if (data.use_compressed_matrix1)
                        for (int i = 0; i < update_indices01.size(); i++)
                            p1_fF_comp.update_elem(
                                update_indices01(i),
                                data.p1_comp_fF.coeff(update_indices01(i), 2 * v) * (fF(2 * v) - fF_old(2 * v))
                            );
                    if (data.use_compressed_matrix2)
                        for (int i = 0; i < update_indices02.size(); i++)
                            p2_fF_comp.update_elem(
                                update_indices02(i),
                                data.p2_comp_fF.coeff(update_indices02(i), 2 * v) * (fF(2 * v) - fF_old(2 * v))
                            );
                    if (data.use_compressed_matrix1)
                        for (int i = 0; i < update_indices11.size(); i++)
                            p1_fF_comp.update_elem(
                                update_indices11(i), data.p1_comp_fF.coeff(update_indices11(i), 2 * v + 1) *
                                                         (fF(2 * v + 1) - fF_old(2 * v + 1))
                            );
                    if (data.use_compressed_matrix2)
                        for (int i = 0; i < update_indices12.size(); i++)
                            p2_fF_comp.update_elem(
                                update_indices12(i), data.p2_comp_fF.coeff(update_indices12(i), 2 * v + 1) *
                                                         (fF(2 * v + 1) - fF_old(2 * v + 1))
                            );
                }

                ScalarType inf_norm = fF.lpNorm<Eigen::Infinity>();
                if (inf_norm == 0.0 || (fF - fF_old).lpNorm<Eigen::Infinity>() / inf_norm < epsilon) break;

                updated = true;
            }

            // ScalarType complementarity_error = 0.0;
            // for (unsigned int v=0; v<num_collisions; v++) {
            //     complementarity_error +=  (A.middleRows<2>(2*v) * fF - b.segment<2>(2*v)).norm() *
            //     std::abs(fF.segment<2>(2*v).norm() - data.friction_coeff * fN(v));
            // }
            // if (complementarity_error > 1e-3)
            //     std::cout<<pow1_idx<<"-"<<pow2_idx<<" iteration "<<i<< " F: "<<
            //     complementarity_error<<std::endl;
        }

        // normal direction
        {
            VectorXs p1 = data.p1_fF * fF + data.p1_f1 * f1, p2 = data.p2_fF * fF + data.p2_f2 * f2;
            VectorXs b = -data.N_p1 * p1 + data.N_p2 * p2 - data.b_N;
            if (data.is_deformable1) {
                if (data.use_compressed_matrix1)
                    b -= data.N_u1_comp * CompressedMatrixVectorProduct<>::rows(
                                              data.u1_p1, CompressedVector<>(data.comp_perm1 * p1), data.N_u1_cols
                                          );
                else
                    b -= data.N_u1_comp * data.u1_p1_uncompressed * p1;
            }
            if (data.is_deformable2) {
                if (data.use_compressed_matrix2)
                    b += data.N_u2_comp * CompressedMatrixVectorProduct<>::rows(
                                              data.u2_p2, CompressedVector<>(data.comp_perm2 * p2), data.N_u2_cols
                                          );
                else
                    b += data.N_u2_comp * data.u2_p2_uncompressed * p2;
            }

            CompressedVector<> p1_fN_comp;
            if (data.use_compressed_matrix1) p1_fN_comp = CompressedVector<>(data.p1_comp_fN * fN);
            CompressedVector<> p2_fN_comp;
            if (data.use_compressed_matrix2) p2_fN_comp = CompressedVector<>(data.p2_comp_fN * fN);

            for (unsigned int j = 0; j < normal_iter; j++) {
                VectorXs fN_old = fN;
                for (int v = 0; v < num_collisions; v++) {
                    auto update_indices1 = data.use_compressed_matrix1 ? get_col_indices(data.N_u1_comp, v)
                                                                       : Eigen::Map<const VectorXi>(nullptr, 0);
                    auto update_indices2 = data.use_compressed_matrix2 ? get_col_indices(data.N_u2_comp, v)
                                                                       : Eigen::Map<const VectorXi>(nullptr, 0);

                    ScalarType b_v = b(v) - data.N_fN.row(v) * fN - diagonal_scaling_factor * data.A_NN_diag(v) * fN(v);
                    if (data.is_deformable1) {
                        if (data.use_compressed_matrix1)
                            b_v -= data.N_u1_comp.row(v) *
                                   CompressedMatrixVectorProduct<>::rows(data.u1_p1, p1_fN_comp, update_indices1);
                        else
                            b_v -= data.N_u1_comp.row(v) * data.u1_p1_uncompressed * data.p1_comp_fN * fN;
                    }
                    if (data.is_deformable2) {
                        if (data.use_compressed_matrix2)
                            b_v += data.N_u2_comp.row(v) *
                                   CompressedMatrixVectorProduct<>::rows(data.u2_p2, p2_fN_comp, update_indices2);
                        else
                            b_v += data.N_u2_comp.row(v) * data.u2_p2_uncompressed * data.p2_comp_fN * fN;
                    }
                    fN(v) += b_v / (data.A_NN_diag(v) * (1. + diagonal_scaling_factor));

                    if (fN(v) < 0.0) fN(v) = 0.0;

                    if (data.use_compressed_matrix1)
                        for (int i = 0; i < update_indices1.size(); i++)
                            p1_fN_comp.update_elem(
                                update_indices1(i), data.p1_comp_fN.coeff(update_indices1(i), v) * (fN(v) - fN_old(v))
                            );
                    if (data.use_compressed_matrix2)
                        for (int i = 0; i < update_indices2.size(); i++)
                            p2_fN_comp.update_elem(
                                update_indices2(i), data.p2_comp_fN.coeff(update_indices2(i), v) * (fN(v) - fN_old(v))
                            );
                }

                ScalarType inf_norm = fN.lpNorm<Eigen::Infinity>();
                if (inf_norm == 0.0 || (fN - fN_old).lpNorm<Eigen::Infinity>() / inf_norm < epsilon) break;

                updated = true;
            }

            // ScalarType complementarity_error = std::abs(fN.transpose() *  (A * fN - b));
            // ScalarType constraints_error = ((A * fN) -b).cwiseMin(0.0).norm();
            // if (complementarity_error > 1e-3 || constraints_error > 1e-3)
            //     std::cout<<pow1_idx<<"-"<<pow2_idx<<" iteration "<<i<< " N: "<< complementarity_error <<
            //     "\t,
            //     "<< constraints_error <<std::endl;
        }
    }

    f1 += data.f1_fN * fN + data.f1_fF * fF;
    f2 += data.f2_fN * fN + data.f2_fF * fF;

    return updated;
}

bool ConstraintsSolver::solve_glue_constraints(ConstraintData &constraint_data) {
    auto get_col_indices = [](const SparseMatrix &mat, int row) {
        auto size = mat.outerIndexPtr()[row + 1] - mat.outerIndexPtr()[row];
        auto ptr = mat.innerIndexPtr() + mat.outerIndexPtr()[row];
        return Eigen::Map<const VectorXi>(ptr, size);
    };

    bool updated = false;

    GlueConstraintData &data = *constraint_data.glue_constraint_data;
    VectorXs &f1 = physics_object_wrappers[constraint_data.object1_idx].f;
    VectorXs &f2 = physics_object_wrappers[constraint_data.object2_idx].f;
    VectorXs &fP = data.fP;

    size_t num_constraints = fP.rows();

    f1 -= data.f1_fP * fP;
    f2 -= data.f2_fP * fP;

    VectorXs p1 = data.p_f1 * f1, p2 = data.p_f2 * f2;
    VectorXs b = -data.P_p1 * p1 + data.P_p2 * p2 - data.b;
    if (data.is_deformable1) {
        if (data.use_compressed_matrix1)
            b -= data.P_u1_comp * CompressedMatrixVectorProduct<>::rows(
                                      data.u1_p1, CompressedVector<>(data.comp_perm1 * p1), data.P_u1_cols
                                  );
        else
            b -= data.P_u1_comp * data.u1_p1_uncompressed * p1;
    }
    if (data.is_deformable2) {
        if (data.use_compressed_matrix2)
            b += data.P_u2_comp * CompressedMatrixVectorProduct<>::rows(
                                      data.u2_p2, CompressedVector<>(data.comp_perm2 * p2), data.P_u2_cols
                                  );
        else
            b += data.P_u2_comp * data.u2_p2_uncompressed * p2;
    }

    CompressedVector<> p1_fP_comp;
    if (data.use_compressed_matrix1) p1_fP_comp = CompressedVector<>(data.p1_comp_fP * fP);
    CompressedVector<> p2_fP_comp;
    if (data.use_compressed_matrix2) p2_fP_comp = CompressedVector<>(data.p2_comp_fP * fP);

    for (unsigned int j = 0; j < glue_iter; j++) {
        VectorXs fP_old = fP;
        for (int v = 0; v < num_constraints; v++) {
            auto update_indices1 = data.use_compressed_matrix1 ? get_col_indices(data.P_u1_comp, v)
                                                               : Eigen::Map<const VectorXi>(nullptr, 0);
            auto update_indices2 = data.use_compressed_matrix2 ? get_col_indices(data.P_u2_comp, v)
                                                               : Eigen::Map<const VectorXi>(nullptr, 0);

            ScalarType b_v = b(v) - data.P_fP.row(v) * fP - diagonal_scaling_factor * data.A_PP_diag(v) * fP(v);
            if (data.is_deformable1) {
                if (data.use_compressed_matrix1)
                    b_v -= data.P_u1_comp.row(v) *
                           CompressedMatrixVectorProduct<>::rows(data.u1_p1, p1_fP_comp, update_indices1);
                else
                    b_v -= data.P_u1_comp.row(v) * data.u1_p1_uncompressed * data.p1_comp_fP * fP;
            }
            if (data.is_deformable2) {
                if (data.use_compressed_matrix2)
                    b_v += data.P_u2_comp.row(v) *
                           CompressedMatrixVectorProduct<>::rows(data.u2_p2, p2_fP_comp, update_indices2);
                else
                    b_v += data.P_u2_comp.row(v) * data.u2_p2_uncompressed * data.p2_comp_fP * fP;
            }
            fP(v) += b_v / (data.A_PP_diag(v) * (1. + diagonal_scaling_factor));

            if (data.use_compressed_matrix1)
                for (int i = 0; i < update_indices1.size(); i++)
                    p1_fP_comp.update_elem(
                        update_indices1(i), data.p1_comp_fP.coeff(update_indices1(i), v) * (fP(v) - fP_old(v))
                    );
            if (data.use_compressed_matrix2)
                for (int i = 0; i < update_indices2.size(); i++)
                    p2_fP_comp.update_elem(
                        update_indices2(i), data.p2_comp_fP.coeff(update_indices2(i), v) * (fP(v) - fP_old(v))
                    );
        }

        ScalarType inf_norm = fP.lpNorm<Eigen::Infinity>();
        if (inf_norm == 0.0 || (fP - fP_old).lpNorm<Eigen::Infinity>() / inf_norm < epsilon) break;

        updated = true;
    }
    f1 += data.f1_fP * fP;
    f2 += data.f2_fP * fP;

    return updated;
}

bool ConstraintsSolver::solve_fixed_pos_constraints(ConstraintData &constraint_data) {
    auto get_col_indices = [](const SparseMatrix &mat, int row) {
        auto size = mat.outerIndexPtr()[row + 1] - mat.outerIndexPtr()[row];
        auto ptr = mat.innerIndexPtr() + mat.outerIndexPtr()[row];
        return Eigen::Map<const VectorXi>(ptr, size);
    };

    bool updated = false;

    FixedPosConstraintData &data = *constraint_data.fixed_pos_constraint_data;
    VectorXs &f = physics_object_wrappers[constraint_data.object1_idx].f;
    VectorXs &fP = data.fP;

    size_t num_constraints = fP.rows();

    f -= data.f_fP * fP;

    VectorXs p = data.p_f * f;
    VectorXs b = -data.P_p * p - data.b;
    if (data.is_deformable) {
        if (data.use_compressed_matrix)
            b -= data.P_u_comp *
                 CompressedMatrixVectorProduct<>::rows(data.u_p, CompressedVector<>(data.comp_perm * p), data.P_u_cols);
        else
            b -= data.P_u_comp * data.u_p_uncompressed * p;
    }

    CompressedVector<> p_fP_comp;
    if (data.use_compressed_matrix) p_fP_comp = CompressedVector<>(data.p_comp_fP * fP);

    for (unsigned int j = 0; j < fixed_pos_iter; j++) {
        VectorXs fP_old = fP;
        for (int v = 0; v < num_constraints; v++) {
            ScalarType b_v = b(v) - data.P_fP.row(v) * fP - diagonal_scaling_factor * data.A_PP_diag(v) * fP(v);

            if (data.use_compressed_matrix) {
                auto update_indices = get_col_indices(data.P_u_comp, v);
                if (data.is_deformable)
                    b_v -= data.P_u_comp.row(v) *
                           CompressedMatrixVectorProduct<>::rows(data.u_p, p_fP_comp, update_indices);
                fP(v) += b_v / (data.A_PP_diag(v) * (1. + diagonal_scaling_factor));

                for (int i = 0; i < update_indices.size(); i++)
                    p_fP_comp.update_elem(
                        update_indices(i), data.p_comp_fP.coeff(update_indices(i), v) * (fP(v) - fP_old(v))
                    );
            } else {
                if (data.is_deformable) b_v -= data.P_u_comp.row(v) * data.u_p_uncompressed * data.p_comp_fP * fP;
                fP(v) += b_v / (data.A_PP_diag(v) * (1. + diagonal_scaling_factor));
            }
        }

        ScalarType inf_norm = fP.lpNorm<Eigen::Infinity>();
        if (inf_norm == 0.0 || (fP - fP_old).lpNorm<Eigen::Infinity>() / inf_norm < epsilon) break;

        updated = true;
    }

    // std::cout<<(b - data.P_fP * fP - data.P_u_comp * (VectorXs)(data.u_p *
    // CompressedVector<>(data.p_comp_fP
    // * fP))).lpNorm<Eigen::Infinity>()<<std::endl;

    f += data.f_fP * fP;

    return updated;
}

void ConstraintsSolver::solve_constraints() {
    for (int i = 0; i < global_iter; i++) {
        std::cout << i << std::endl;
        volatile bool updated = false;

        for (auto &batch : constraints_solution_batches) {
#pragma omp parallel for
            for (ConstraintData &constraint : batch) {
                bool _updated;

                switch (constraint.constraint_type) {
                case COLLISION: _updated = solve_collision_constraints(constraint); break;
                case GLUE: _updated = solve_glue_constraints(constraint); break;
                case FIXED_POS: _updated = solve_fixed_pos_constraints(constraint); break;
                }

                if (_updated) updated = true;
            }
        }

        if (!updated) break;
    }

    for (ConstraintData &data : constraints_data_vector) {
        PhysicsObjectWrapper &pow1 = physics_object_wrappers[data.object1_idx];
        PhysicsBaseObject &po1 = pow1.physics_object;
        po1.p = po1.p_f * physics_object_wrappers[data.object1_idx].f;

        if (!data.is_single_object_contraint) {
            PhysicsObjectWrapper &pow2 = physics_object_wrappers[data.object2_idx];
            PhysicsBaseObject &po2 = pow2.physics_object;
            po2.p = po2.p_f * physics_object_wrappers[data.object2_idx].f;
        }
    }
}
