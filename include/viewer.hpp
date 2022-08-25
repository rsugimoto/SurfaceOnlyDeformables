#ifndef __VIEWER_HPP__
#define __VIEWER_HPP__

#include "json.hpp"
#include "type_declaration.hpp"
#include <vector>

#define IGL_VIEWER_VIEWER_QUIET
#include <igl/opengl/glfw/Viewer.h>

class Viewer {
  public:
    struct MeshInitData {
        const MatrixX3s &V;
        const MatrixX3i &F;
        const Vector3s translation;
        const Matrix3s rotation;
        bool wireframe;
    };

    struct MeshUpdateData {
        const MatrixX3s &V;
        const Vector3s translation;
        const Matrix3s rotation;
        const VectorXs normalized_p;
        size_t index;
    };

    Viewer();
    ~Viewer();

    bool init(const nlohmann::json &config, std::vector<MeshInitData> &init_list);
    bool update(std::vector<MeshUpdateData> &update_list);
    bool loop();

    volatile bool terminated;

  private:
    igl::opengl::glfw::Viewer viewer;

    volatile bool update_processed;
    volatile bool frame_saved;
    std::vector<MeshUpdateData> update_list;

    bool save_frames;
    std::string frame_folder_path;
    unsigned int frame_number;
    std::vector<Eigen::RowVector3d> colors;
    std::vector<Eigen::Vector3d> bbs;
};

#endif //!__VIEWER_HPP__