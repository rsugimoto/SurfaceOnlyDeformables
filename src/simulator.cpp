#include "simulator.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "json.hpp"

#include "elastodynamics_cqbem_matrix_collection.hpp"
#include "elastodynamics_drbem_matrix_collection.hpp"
#include "elastostatics_bem_matrix_collection.hpp"
#include "physics_base_matrix_collection.hpp"
#include "rigidbody_matrix_collection.hpp"

#include "elastodynamics_cqbem_object.hpp"
#include "elastodynamics_drbem_object.hpp"
#include "elastostatics_bem_object.hpp"
#include "rigidbody_object.hpp"

#include "constraints_solver.hpp"
#include "matrix_io.hpp"
#include "timer.hpp"
#include "viewer.hpp"

#ifdef OMP_AVAILABLE
#include <omp.h>
#endif

Simulator::Simulator()
    : gravitational_constant(0.0, -9.8, 0.0), t(0.0), terminated(false), enable_cuda(true), enable_viewer(true),
      save_mesh(false), cuda_thread_per_block(32), time_step(0), end_time_step(-1) {}
Simulator::~Simulator() {}

using json = nlohmann::json;

bool Simulator::init(int argc, char *argv[]) {
    json config;
    {
        std::ifstream config_file(argv[1]);
        if (!config_file) {
            std::cout << "Failed to load config file: " << argv[1] << std::endl;
            return false;
        }
        config_file >> config;
        config_file.close();
    }

    dt = config["dt"].get<ScalarType>();
    if (config.contains("gravitational_constant")) {
        for (int i = 0; i < 3; i++) gravitational_constant(i) = config["gravitational_constant"][i].get<ScalarType>();
    }

    if (config.contains("end_time_step")) end_time_step = config["end_time_step"].get<unsigned int>();

    if (config.contains("enable_cuda")) enable_cuda = config["enable_cuda"].get<bool>();
#ifndef CUDA_AVAILABLE
    if (enable_cuda) std::cout << "The program is not compiled with CUDA. CUDA is disabled." << std::endl;
    enable_cuda = false;
#endif

    if (config.contains("enable_viewer")) enable_viewer = config["enable_viewer"].get<bool>();

    if (config.contains("save_mesh")) save_mesh = config["save_mesh"].get<bool>();

    if (save_mesh) {
        mesh_save_folder_path = config["mesh_save_folder_path"].get<std::string>();
        std::filesystem::create_directories(mesh_save_folder_path);
    }

    if (config.contains("cuda_thread_per_block"))
        cuda_thread_per_block = config["cuda_thread_per_block"].get<unsigned int>();

#ifdef OMP_AVAILABLE
    if (config.contains("omp_num_threads")) omp_set_num_threads(config["omp_num_threads"].get<int>());
#endif

    std::vector<Viewer::MeshInitData> mesh_init_data;
    for (auto &physics_object_groups_config : config["physics_object_groups"]) {
        json physics_object_config;
        {
            if (physics_object_groups_config["physics_object_group"].is_string()) {
                std::string physics_object_config_file_path =
                    physics_object_groups_config["physics_object_group"].get<std::string>();
                std::ifstream config_file(physics_object_config_file_path);
                if (!config_file) return false;
                config_file >> physics_object_config;
                config_file.close();
            } else if (physics_object_groups_config["physics_object_group"].is_object()) {
                physics_object_config = physics_object_groups_config["physics_object_group"];
            } else
                return 0;
        }

        std::shared_ptr<PhysicsBaseMatrixCollection> physics_matrix_collection;
        std::string physics_object_type = physics_object_config["type"];
        if (physics_object_type == "elastodynamicsCQBEM")
            physics_matrix_collection =
                std::make_shared<ElastodynamicsCQBEMMatrixCollection>(dt, enable_cuda, cuda_thread_per_block);

        else if (physics_object_type == "elastodynamicsDRBEM")
            physics_matrix_collection = std::make_shared<ElastodynamicsDRBEMMatrixCollection>(dt);
        else if (physics_object_type == "elastostaticsBEM")
            physics_matrix_collection = std::make_shared<ElastostaticsBEMMatrixCollection>();
        else if (physics_object_type == "rigidbody")
            physics_matrix_collection = std::make_shared<RigidbodyMatrixCollection>();
        else {
            std::cout << "Invalid physics object group type detected. Valid types are "
                         "\"elastodynamicsCQBEM\"/\"elastodynamicsDRBEM\"/"
                         "\"elastostaticsBEM\"/\"rigidbody\""
                      << std::endl;
            return false;
        }

        if (!physics_matrix_collection->init(physics_object_config)) return false;

        for (auto &physics_object_instance_config : physics_object_groups_config["physics_objects"]) {
            if (physics_object_type == "elastodynamicsCQBEM")
                physics_objects.emplace_back(std::make_unique<ElastodynamicsCQBEMObject>(
                    std::static_pointer_cast<ElastodynamicsCQBEMMatrixCollection>(physics_matrix_collection), dt,
                    gravitational_constant
                ));
            else if (physics_object_type == "elastodynamicsDRBEM")
                physics_objects.emplace_back(std::make_unique<ElastodynamicsDRBEMObject>(
                    std::static_pointer_cast<ElastodynamicsDRBEMMatrixCollection>(physics_matrix_collection), dt,
                    gravitational_constant
                ));
            else if (physics_object_type == "elastostaticsBEM")
                physics_objects.emplace_back(std::make_unique<ElastostaticsBEMObject>(
                    std::static_pointer_cast<ElastostaticsBEMMatrixCollection>(physics_matrix_collection), dt,
                    gravitational_constant
                ));
            else if (physics_object_type == "rigidbody")
                physics_objects.emplace_back(std::make_unique<RigidbodyObject>(
                    std::static_pointer_cast<RigidbodyMatrixCollection>(physics_matrix_collection), dt,
                    gravitational_constant
                ));

            if (!physics_objects.back()->init(physics_object_instance_config)) return false;

            const PhysicsBaseObject &po = *physics_objects.back();
            mesh_init_data.push_back(
                {po.get_V_global(), po.get_F(), po.get_translation(), po.get_rotation(), po.wireframe}
            );
        }
    }
    constraints_solver = std::make_unique<ConstraintsSolver>();
    if (!constraints_solver->init(config["constraints"], physics_objects)) return false;
    if (enable_viewer) {
        viewer = std::make_unique<Viewer>();
        if (!viewer->init(config["viewer"], mesh_init_data)) return false;
    }

    timer = std::make_unique<Timer>();
    if (config.contains("timer_stat_save_path"))
        timer->set_stat_save_path(config["timer_stat_save_path"].get<std::string>());

    return true;
}

void Simulator::step() {
    auto d1 = timer->measure("1. unconstrained motion", [&] {
#pragma omp parallel for
        for (size_t i = 0; i < physics_objects.size(); i++) physics_objects[i]->estimate_next_state();
    });

    auto d2 = timer->measure("2. collision detection", [&] {
        constraints_solver->update_meshes();
        constraints_solver->detect_collision();
    });

    auto d3 = timer->measure("3. constraints solve precomputation", [&] { constraints_solver->precompute(); });

    auto d4 = timer->measure("4. constraints solve", [&] { constraints_solver->solve_constraints(); });

    auto d5 = timer->measure("5. drift elimination & frame updates", [&] {
#pragma omp parallel for
        for (size_t i = 0; i < physics_objects.size(); i++) physics_objects[i]->confirm_next_state();
    });

    auto d6 = timer->measure("6. viewer update", [&] {
        std::vector<Viewer::MeshUpdateData> mesh_update_data;
        unsigned int po_index = 0;
        for (auto &po : physics_objects) {
            if (!(po->is_fixed && !po->is_deformable))
                mesh_update_data.push_back(
                    {po->get_V_global(), po->get_translation(), po->get_rotation(), po->get_normalized_p(), po_index}
                );
            po_index++;
        }
        if (enable_viewer) viewer->update(mesh_update_data);
    });

    auto d7 = timer->measure("7. save mesh", [&] {
        if (!save_mesh) return;
        int po_index = 0;
        for (auto &po : physics_objects) {
            volatile bool init_end = false;
            std::thread save_mesh_thread([&]() {
                MatrixX3s V = po->get_permutation_matrix() * (po->get_V_local().rowwise() + po->get_cm());
                std::string obj_file_name = mesh_save_folder_path + "/obj" + std::to_string(po_index) + "_frame" +
                                            std::to_string(time_step) + ".mat";
                std::string json_file_name = mesh_save_folder_path + "/obj" + std::to_string(po_index) + "_frame" +
                                             std::to_string(time_step) + ".json";

                std::array<ScalarType, 9> rot;
                std::array<ScalarType, 3> trans;
                Vector3s _trans = po->get_translation() - (po->get_rotation() * (po->get_cm().transpose())).eval();
                for (int i = 0; i < 3; i++) {
                    trans[i] = _trans(i);
                    for (int j = 0; j < 3; j++) { rot[3 * i + j] = po->get_rotation()(i, j); }
                }

                init_end = true;
                if (po->is_traction_discontinuity_enabled()) {
                    V.conservativeResize(po->get_original_num_vertices(), 3);
                }
                Eigen::save_matrix(V, obj_file_name);

                json data;
                data["rotation"] = rot;
                data["translation"] = trans;

                std::ofstream out(json_file_name.c_str(), std::ios::out);
                out << data.dump(4);
                out.close();
            });
            while (!init_end)
                ;
            save_mesh_thread.detach();
            po_index++;
        }
    });

    t += dt;
    std::cout << ++time_step << ": t = " << t << " ("
              << std::chrono::duration_cast<std::chrono::milliseconds>(d1).count() << ", "
              << std::chrono::duration_cast<std::chrono::milliseconds>(d2).count() << ", "
              << std::chrono::duration_cast<std::chrono::milliseconds>(d3).count() << ", "
              << std::chrono::duration_cast<std::chrono::milliseconds>(d4).count() << ", "
              << std::chrono::duration_cast<std::chrono::milliseconds>(d5).count() << ", "
              << std::chrono::duration_cast<std::chrono::milliseconds>(d6).count() << ", "
              << std::chrono::duration_cast<std::chrono::milliseconds>(d7).count() << ")" << std::endl;

    if ((enable_viewer && viewer->terminated) || time_step >= end_time_step) terminated = true;
}

bool Simulator::loop() {
    if (enable_viewer) {
        auto simulation_thread = std::thread([&] {
            while (!terminated) step();
        });
        viewer->loop();
        simulation_thread.join();
    } else {
        while (!terminated) step();
    }

    if (timer.get() != nullptr) timer->display_average_times<std::chrono::microseconds>();

    return true;
}