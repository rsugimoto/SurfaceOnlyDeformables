#include "physics_base_matrix_collection.hpp"

#include <iostream>
#include <map>
#include <set>
#include <utility> //pair
#include <vector>

#include <igl/adjacency_list.h>
#include <igl/centroid.h>
#include <igl/decimate.h>
#include <igl/doublearea.h>
#include <igl/false_barycentric_subdivision.h>
#include <igl/loop.h>
#include <igl/per_face_normals.h>
#include <igl/read_triangle_mesh.h>
#include <igl/sharp_edges.h>
#include <igl/upsample.h>
#include <igl/write_triangle_mesh.h>

#ifdef OPENCCL_AVAILABLE
#include "OpenCCL.h"
#endif

#include "cross_product_matrix.hpp"
#include "integrator_collocation.hpp"

enum SubdivisionScheme { Upsample, Loop, FalseBarycentric };

PhysicsBaseMatrixCollection::PhysicsBaseMatrixCollection() : use_compressed_matrices(false){};

bool PhysicsBaseMatrixCollection::init(const nlohmann::json &config) {
    if (!load_mesh(config["mesh"])) return false;

    rho = config["material"]["rho"].get<ScalarType>();

    MatrixX3s N;
    igl::per_face_normals(V, F, N);

    {
        igl::centroid(V, F, cm, mass);
        mass *= rho;
        original_cm = cm;
        V.rowwise() -= cm;
        cm.setZero();
        I = compute_inertia_tensor(N);
    }

    translational_acc_mat = compute_external_translational_acc_matrix(mass);
    rotational_acc_mat = compute_external_rotational_acc_matrix(I);

    compression_permutation.setIdentity(3 * num_vertices);
    u_p_compressed = CompressedMatrix<>(3 * num_vertices);
    u_p_diag.resize(3 * num_vertices, 3 * num_vertices);

    return true;
}

bool PhysicsBaseMatrixCollection::load_mesh(const nlohmann::json &config) {
    const std::string filename = config["file"].get<std::string>();
    std::cout << "file:" << filename << std::endl;
    int subdivision_level = config.contains("subdivision_level") ? config["subdivision_level"].get<int>() : 0;
    const std::string subdivision_scheme =
        config.contains("subdivision_scheme") ? config["subdivision_scheme"].get<std::string>() : "Upsample";
    double decimation_scale = config.contains("decimation_scale") ? config["decimation_scale"].get<double>() : 1.0;
    double scale = config.contains("scale") ? config["scale"].get<double>() : 1.0;

    enable_traction_discontinuity =
        config.contains("enable_traction_discontinuity") ? config["enable_traction_discontinuity"].get<bool>() : false;
    double sharp_edge_threshold_degree =
        config.contains("sharp_edge_threshold_degree") ? config["sharp_edge_threshold_degree"].get<double>() : 20.0;

    if (!igl::read_triangle_mesh(filename, V, F)) return false;

    // Remove unused vertices
    {
        std::map<int, int> vertices;
        for (Eigen::Index f = 0; f < F.rows(); f++) {
            auto face = F.row(f);
            vertices.emplace(face[0], 0);
            vertices.emplace(face[1], 0);
            vertices.emplace(face[2], 0);
        }

        if (vertices.size() != V.rows()) {
            MatrixX3s V_new(vertices.size(), 3);
            {
                size_t i = 0;
                {
                    for (auto &[v_old, v_new] : vertices) {
                        v_new = i++;
                        V_new.row(v_new) = V.row(v_old);
                    }
                }
            }
            MatrixX3i F_new(F.rows(), 3);
            for (Eigen::Index f = 0; f < F.rows(); f++) {
                auto face = F.row(f);
                F_new.row(f) = RowVector3i(vertices[face[0]], vertices[face[1]], vertices[face[2]]);
            }

            std::cout << "Removed unused vertices: " << V.rows() << " -> " << V_new.rows() << std::endl;
            V.swap(V_new);
            F.swap(F_new);
        }
    }
    V = scale * V;

    // subdivision
    {
        SubdivisionScheme scheme;
        if (subdivision_scheme == "Upsample")
            scheme = Upsample;
        else if (subdivision_scheme == "Loop")
            scheme = Loop;
        else if (subdivision_scheme == "FalseBarycentric")
            scheme = FalseBarycentric;
        else
            scheme = Upsample;

        for (auto i = 0; i < subdivision_level; ++i) {
            if (scheme == Upsample || scheme == Loop) {
                Eigen::SparseMatrix<ScalarType> S;
                MatrixX3i NF;
                switch (scheme) {
                case Upsample: igl::upsample(V.rows(), F, S, NF); break;
                case Loop: igl::loop(V.rows(), F, S, NF); break;
                default: break;
                }
                F = NF;
                V = S * V;
            } else {
                Eigen::MatrixXd _V = V.cast<double>();
                Eigen::MatrixXd VD;
                MatrixX3i FD;
                igl::false_barycentric_subdivision(_V, F, VD, FD);
                V = VD.cast<ScalarType>();
                F = FD;
            }
        }
    }

    // decimation
    if (decimation_scale < 1.0) {
        Eigen::MatrixXd U;
        Eigen::MatrixXi G;
        VectorXi J;
        igl::decimate(V.cast<double>(), F, (Eigen::Index)(F.rows() * decimation_scale), U, G, J);
        V = U.cast<ScalarType>();
        F = G;
    }

    // flip normal
    if (config.contains("flip_normal") && config["flip_normal"].get<bool>()) F.col(1).swap(F.col(2));

    perm.resize(V.rows());
    perm.setIdentity();

    // Reorder vertices (or don't)
    bool enable_vertex_reordering;
    {

        enable_vertex_reordering = true;
        if (config.contains("enable_vertex_reordering"))
            enable_vertex_reordering = config["enable_vertex_reordering"].get<bool>();
#ifndef OPENCCL_AVAILABLE
        if (enable_vertex_reordering)
            std::cout << "Program not compiled with OpenCCL enabled. Vertex reordering is disabled." << std::endl;
        enable_vertex_reordering = false;
#endif

        if (enable_vertex_reordering) {
#ifdef OPENCCL_AVAILABLE
            OpenCCL::CLayoutGraph graph(V.rows());
            for (Eigen::Index f = 0; f < F.rows(); f++) {
                graph.AddEdge(F(f, 0), F(f, 1));
                graph.AddEdge(F(f, 1), F(f, 2));
                graph.AddEdge(F(f, 2), F(f, 0));
            }
            graph.ComputeOrdering(perm.indices().data());
#endif

            // If vertices are reordered, save the mesh with reordering.
            Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm_trans = perm.transpose();
            MatrixX3s _V = perm_trans * V;
            MatrixX3i _F = F;
            for (Eigen::Index f = 0; f < F.rows(); f++) {
                _F(f, 0) = perm_trans.indices()(F(f, 0));
                _F(f, 1) = perm_trans.indices()(F(f, 1));
                _F(f, 2) = perm_trans.indices()(F(f, 2));
            }
            igl::write_triangle_mesh(filename.substr(0, filename.size() - 4) + "_reordered.obj", _V, _F);

            if (!enable_traction_discontinuity) {
                V = _V;
                F = _F;
            }
        }
    }

    original_num_vertices = V.rows();

    // duplicate vertices for traction discontinuity. This operates on the mesh without vertex reordering first, and
    // consider the permutation due to vertex reordering later.
    if (enable_traction_discontinuity) {
        // save the original input matrices
        original_F = F;
        original_V = V;

        std::map<Eigen::Index, std::vector<Eigen::Index>> vertex_map; // map from old vert index to new vert indices

        VectorXi EMAP;
        MatrixX2i SE, E, uE;
        std::vector<std::vector<Eigen::Index>> uE2E;
        std::vector<Eigen::Index> sharp;

        // Eigen::MatrixXi E, uE;
        igl::sharp_edges(V, F, sharp_edge_threshold_degree * M_PI / 180.0, SE, E, uE, EMAP, uE2E, sharp);

        // get a set of directed sharp edges
        std::map<std::pair<Eigen::Index, Eigen::Index>, Eigen::Index> directed_sharp_edges;
        for (Eigen::Index e = 0; e < sharp.size(); e++) {
            assert(uE2E[sharp[e]].size() == 2);
            assert(
                E(uE2E[sharp[e]][0], 1) == E(uE2E[sharp[e]][1], 0) && E(uE2E[sharp[e]][0], 0) == E(uE2E[sharp[e]][1], 1)
            );
            directed_sharp_edges[{E(uE2E[sharp[e]][0], 1), E(uE2E[sharp[e]][0], 0)}] = uE2E[sharp[e]][0];
            directed_sharp_edges[{E(uE2E[sharp[e]][1], 1), E(uE2E[sharp[e]][1], 0)}] = uE2E[sharp[e]][1];
        }

        std::map<std::pair<Eigen::Index, Eigen::Index>, Eigen::Index> directed_edges;
        for (Eigen::Index e = 0; e < E.rows(); e++) directed_edges[{E(e, 1), E(e, 0)}] = e;

        // get a sorted adjacency list
        std::vector<std::vector<Eigen::Index>> A;
        igl::adjacency_list(F, A, true);

        // update F with new vertex indices
        Eigen::Index new_vertex_index = V.rows();

        while (!directed_sharp_edges.empty()) {
            auto curr_iter = directed_sharp_edges.begin();
            if (vertex_map.count((*curr_iter).first.first)) {
                vertex_map[curr_iter->first.first].emplace_back(new_vertex_index++);
            } else {
                vertex_map.emplace((*curr_iter).first.first, std::vector<Eigen::Index>());
                vertex_map[curr_iter->first.first].emplace_back(curr_iter->first.first);
            };
            // std::cout<<curr_iter->first.first<<"("<<*(vertex_map[curr_iter->first.first].end()-1)<<")";
            Eigen::Index start_vert_index = curr_iter->first.first;
            Eigen::Index start_vert_index2 = curr_iter->first.second;
            while (curr_iter != directed_sharp_edges.end()) { // loop for each sharp-edge loop
                auto curr = *curr_iter;
                directed_sharp_edges.erase(curr_iter);

                auto &curr_adj_list = A[curr.first.second];
                int first_pos = 0;
                while (curr_adj_list[first_pos] != curr.first.first) first_pos++;

                int pos = first_pos;
                // find the next vertex in the one-ring structure
                do {
                    pos = (pos + 1) % curr_adj_list.size();

                    curr_iter = directed_sharp_edges.find({curr.first.second, curr_adj_list[pos]});
                    if (curr_iter != directed_sharp_edges.end()) break; // if a sharp edge is found in one-ring, break

                } while (pos != first_pos);

                bool revisiting_first_vert = false;
                if (curr.first.second == start_vert_index) {
                    revisiting_first_vert = true; // terminate one-ring iteration if it goes back to the start vertex
                    curr_iter = directed_sharp_edges.end();
                }

                if (curr_iter != directed_sharp_edges.end()) { // if we continue sharp-edge loop
                    if (vertex_map.count(curr.first.second)) {
                        vertex_map[curr.first.second].emplace_back(new_vertex_index++);
                    } else {
                        vertex_map.emplace(curr.first.second, std::vector<Eigen::Index>());
                        vertex_map[curr.first.second].emplace_back(curr.first.second);
                    };
                } // otherwise, the first vertex already has a vertex index assigned, so no need to assign another one.

                auto edge = curr.second;
                F(edge % (F.rows()), (edge / F.rows() + 2) % 3) = *(vertex_map[curr.first.first].end() - 1);
                F(edge % (F.rows()), (edge / F.rows() + 1) % 3) = *(vertex_map[curr.first.second].end() - 1);

                // ressign the vertex indices to the vertices visited while one-ring iterations
                if (curr_iter != directed_sharp_edges.end()) {
                    int _pos = first_pos;
                    // find the next vertex in the one-ring structure
                    while (pos != _pos) {
                        _pos = (_pos + 1) % curr_adj_list.size();
                        auto edge = directed_edges[{curr.first.second, curr_adj_list[_pos]}];
                        F(edge % (F.rows()), (edge / F.rows() + 2) % 3) = *(vertex_map[curr.first.second].end() - 1);
                    }
                }

                // if we are revisiting the first vertex, we must treat it differently because the first edge is already
                // removed from the directed edges map.
                if (revisiting_first_vert) {
                    int pos = 0;
                    while (curr_adj_list[pos] != start_vert_index2) pos++;
                    int _pos = first_pos;
                    // find the next vertex in the one-ring structure
                    while (pos != _pos) {
                        _pos = (_pos + 1) % curr_adj_list.size();
                        auto edge = directed_edges[{curr.first.second, curr_adj_list[_pos]}];
                        F(edge % (F.rows()), (edge / F.rows() + 2) % 3) = *(vertex_map[curr.first.second].end() - 1);
                    }
                }

                // std::cout<<"-"<<curr.first.second<<"("<<*(vertex_map[curr.first.second].end()-1)<<")";
            }
            // std::cout<<std::endl;
        }

        // add the new vertices to V
        V.conservativeResize(new_vertex_index, 3);
        for (Eigen::Index v = 0; v < original_num_vertices; v++) {
            if (vertex_map.count(v) == 0) {
                vertex_map.emplace(v, std::vector<Eigen::Index>());
                vertex_map[v].emplace_back(v);
            } else {
                for (auto new_index : vertex_map[v]) {
                    if (new_index != v) V.row(new_index) = V.row(v);
                }
            }
        }

        vertex_map_inverse.resize(new_vertex_index);
        for (Eigen::Index v = 0; v < original_num_vertices; v++) {
            for (auto new_index : vertex_map[v]) vertex_map_inverse[new_index] = v;
        }

        // update permutation matrix.
        {
            Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm_new(V.rows());
            Eigen::Index offset = 0;
            for (Eigen::Index v = 0; v < original_V.rows(); v++) {
                for (Eigen::Index j = 0; j < vertex_map[perm.indices()[v]].size(); j++) {
                    auto w = vertex_map[perm.indices()[v]][j];
                    if (j != 0) offset++;
                    perm_new.indices()[v + offset] = w;
                }
            }
            perm = perm_new;
        }
        vertex_map_inverse = perm.transpose() * vertex_map_inverse;

        // If vertices are inserted due to traction discontinuity, update V and F.
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm_trans = perm.transpose();
        V = perm_trans * V;
        for (Eigen::Index f = 0; f < F.rows(); f++) {
            F(f, 0) = perm_trans.indices()(F(f, 0));
            F(f, 1) = perm_trans.indices()(F(f, 1));
            F(f, 2) = perm_trans.indices()(F(f, 2));
        }
        igl::write_triangle_mesh(
            filename.substr(0, filename.size() - 4) + (enable_vertex_reordering ? "_reordered" : "") +
                "_discontinuous.obj",
            V, F
        );
    }

    std::cout << "|V|: " << original_num_vertices << "(" << V.rows() << ")" << std::endl
              << "|F|: " << F.rows() << std::endl;

    num_vertices = V.rows();
    return true;
}

Matrix3s PhysicsBaseMatrixCollection::compute_inertia_tensor(const MatrixX3s &N) {
    RowVector3s cm;
    igl::centroid(V, F, cm);

    Matrix3s I = Matrix3s::Zero();
#pragma omp parallel for
    for (Eigen::Index f_index = 0; f_index < F.rows(); f_index++) {
        const auto n = N.row(f_index);

        const Eigen::Index j1 = F(f_index, 0), j2 = F(f_index, 1), j3 = F(f_index, 2);
        const auto y1 = V.row(j1), y2 = V.row(j2), y3 = V.row(j3);
        Matrix3s mat = gaussian_quadrature_triangle(
            [&](const auto &y) -> Matrix3s {
                RowVector3s r = y - cm;
                Matrix3s res;
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        if (i == j)
                            res(i, j) = (rho / 3.) * (std::pow(r((i + 1) % 3), 3) * n((i + 1) % 3) +
                                                      std::pow(r((i + 2) % 3), 3) * n((i + 2) % 3));
                        else
                            res(i, j) = (-rho / 4.) * (r(j) * r(i) * r(i) * n(i) + r(i) * r(j) * r(j) * n(j));
                    }
                }
                return res;
            },
            y1, y2, y3, 4
        );
#pragma omp critical
        { I += mat; }
    }
    return I;
}

Matrix3Xs PhysicsBaseMatrixCollection::compute_external_translational_acc_matrix(const ScalarType mass) {
    VectorXs A;
    igl::doublearea(V, F, A);

    Matrix3Xs a = Matrix3Xs::Zero(3, 3 * num_vertices);
    for (Eigen::Index f_index = 0; f_index < F.rows(); f_index++) {
        const Eigen::Index j1 = F(f_index, 0), j2 = F(f_index, 1), j3 = F(f_index, 2);
        a.block<3, 3>(0, 3 * j1).diagonal().array() += A(f_index) / 6.;
        a.block<3, 3>(0, 3 * j2).diagonal().array() += A(f_index) / 6.;
        a.block<3, 3>(0, 3 * j3).diagonal().array() += A(f_index) / 6.;
    }
    a = a * (1. / mass);
    return a;
}

Matrix3Xs PhysicsBaseMatrixCollection::compute_external_rotational_acc_matrix(const Matrix3s &I) {
    // Assumes centroid is (0, 0, 0)

    Matrix3Xs a = Matrix3Xs::Zero(3, 3 * num_vertices);

    VectorXs A;
    igl::doublearea(V, F, A);

    for (Eigen::Index f = 0; f < F.rows(); f++) {
        for (int i = 0; i < 3; i++) {
            a.middleCols<3>(3 * F(f, i)) +=
                A(f) / 6. * 0.25 *
                cross_product_matrix(V.row(F(f, 0)) + V.row(F(f, 1)) + V.row(F(f, 2)) + V.row(F(f, i)));
        }
    }

    a = I.inverse() * a;
    return a;
}