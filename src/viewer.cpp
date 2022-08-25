#include "viewer.hpp"

#include <filesystem>
#include <memory>
#include <string>

#include <Eigen/Core>

#include <igl/png/writePNG.h>

Viewer::Viewer() : terminated(false), update_processed(true), frame_number(0) {}
Viewer::~Viewer() {
    while (!terminated)
        ;
}

void update_bounding_box(
    igl::opengl::ViewerData &data, const Vector3s &translation, const Matrix3s &rotation, const Vector3s &size
) {
    Eigen::MatrixXd P(8, 3);
    Eigen::MatrixXi E(12, 2);
    Eigen::RowVector3d C(1., 0., 0.);

    P.row(0) = translation + rotation * Vector3s(size(0) / 2., size(1) / 2., size(2) / 2.);
    P.row(1) = translation + rotation * Vector3s(size(0) / 2., size(1) / 2., -size(2) / 2.);
    P.row(2) = translation + rotation * Vector3s(size(0) / 2., -size(1) / 2., size(2) / 2.);
    P.row(3) = translation + rotation * Vector3s(size(0) / 2., -size(1) / 2., -size(2) / 2.);
    P.row(4) = translation + rotation * Vector3s(-size(0) / 2., size(1) / 2., size(2) / 2.);
    P.row(5) = translation + rotation * Vector3s(-size(0) / 2., size(1) / 2., -size(2) / 2.);
    P.row(6) = translation + rotation * Vector3s(-size(0) / 2., -size(1) / 2., size(2) / 2.);
    P.row(7) = translation + rotation * Vector3s(-size(0) / 2., -size(1) / 2., -size(2) / 2.);

    E.row(0) = Eigen::RowVector2i(0, 1);
    E.row(1) = Eigen::RowVector2i(0, 2);
    E.row(2) = Eigen::RowVector2i(0, 4);
    E.row(3) = Eigen::RowVector2i(1, 3);
    E.row(4) = Eigen::RowVector2i(1, 5);
    E.row(5) = Eigen::RowVector2i(2, 3);
    E.row(6) = Eigen::RowVector2i(2, 6);
    E.row(7) = Eigen::RowVector2i(3, 7);
    E.row(8) = Eigen::RowVector2i(4, 5);
    E.row(9) = Eigen::RowVector2i(4, 6);
    E.row(10) = Eigen::RowVector2i(5, 7);
    E.row(11) = Eigen::RowVector2i(6, 7);

    data.set_edges(P, E, C);
}

bool Viewer::init(const nlohmann::json &config, std::vector<MeshInitData> &init_list) {
    save_frames = config["save_frames"].get<bool>();
    if (save_frames) {
        frame_folder_path = config["frame_folder_path"].get<std::string>();
        std::filesystem::create_directories(frame_folder_path);
    }

    viewer.core().is_animating = true;
    viewer.core().animation_max_fps = 30.0;

    bool is_first_mesh = true;
    for (auto &init_data : init_list) {
        if (!is_first_mesh)
            viewer.append_mesh(true);
        else
            is_first_mesh = false;
        viewer.data().set_mesh(init_data.V.cast<double>(), init_data.F);
        viewer.data().show_lines = true;
        viewer.data().show_faces =
            (config.contains("wireframe") && config["wireframe"].get<bool>()) || !init_data.wireframe;

        colors.push_back({1.0, 0.0, 0.0}); // colors for traction visualization
        colors.push_back(0.1 * Eigen::RowVector3d::Random().array() + 0.9);
        viewer.data().set_colors(colors.back());

        bbs.push_back(
            ((init_data.V.rowwise() - init_data.translation.transpose()) * init_data.rotation).colwise().maxCoeff() -
            ((init_data.V.rowwise() - init_data.translation.transpose()) * init_data.rotation).colwise().minCoeff()
        );
        bbs.back().array() += 1e-3;
        update_bounding_box(viewer.data(), init_data.translation, init_data.rotation, bbs.back());
    }

    if (config.contains("camera")) {
        if (config["camera"].contains("translation")) {
            Eigen::Vector3f camera_translation;
            for (int i = 0; i < 3; i++) camera_translation(i) = config["camera"]["translation"][i].get<float>();
            Eigen::Vector3f translation_offset = init_list.back().V.colwise().mean().cast<float>();
            viewer.core().camera_translation = -camera_translation + translation_offset;
        }
        if (config["camera"].contains("zoom")) viewer.core().camera_zoom = config["camera"]["zoom"].get<float>();
    }

    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &viewer) -> bool {
        if (!update_processed) {
            for (auto &update_data : update_list) {
                Eigen::MatrixXd V = update_data.V.cast<double>();
                viewer.data_list[update_data.index].set_vertices(V);
                Eigen::MatrixXd color = (update_data.normalized_p.cast<double>() *
                                         (colors[2 * update_data.index] - colors[2 * update_data.index + 1]))
                                            .rowwise() +
                                        colors[2 * update_data.index + 1];
                viewer.data_list[update_data.index].set_colors(color);
                viewer.data_list[update_data.index].compute_normals();
                update_bounding_box(
                    viewer.data_list[update_data.index], update_data.translation, update_data.rotation,
                    bbs[update_data.index]
                );
            }
            update_processed = true;
        }
        return false;
    };

    if (save_frames) {
        viewer.callback_post_draw = [&](igl::opengl::glfw::Viewer &viewer) -> bool {
            if (!frame_saved) {
                const int width = viewer.core().viewport(2);
                const int height = viewer.core().viewport(3);

                std::unique_ptr<GLubyte[]> pixels(new GLubyte[width * height * 4]);
                glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.get());

                frame_saved = true;

                std::thread(
                    [&](std::string path, std::unique_ptr<GLubyte[]> pixels, int width, int height) {
                        stbi_write_png(
                            path.c_str(), width, height, 4, pixels.get() + width * (height - 1) * 4, -width * 4
                        );
                    },
                    frame_folder_path + "/F_" + std::to_string(frame_number++) + ".png", std::move(pixels), width,
                    height
                )
                    .detach();
            }
            return false;
        };
    }

    viewer.launch_init(true, false, "Surface-Only Dynamic Deformables", 0, 0);
    return true;
}

bool Viewer::update(std::vector<MeshUpdateData> &update_list) {
    if (update_processed) {
        this->update_list = std::move(update_list);
        update_processed = false;

        if (save_frames) {
            frame_saved = false;
            while (!frame_saved && !terminated)
                ;
        }
    }
    return true;
}

bool Viewer::loop() {
    viewer.launch_rendering(true);
    viewer.launch_shut();
    terminated = true;
    return true;
}