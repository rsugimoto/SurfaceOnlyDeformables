#ifndef __SIMULATOR__
#define __SIMULATOR__

#include <memory>
#include <string>
#include <vector>

#include "type_declaration.hpp"

class PhysicsBaseMatrixCollection;
class PhysicsBaseObject;
class ConstraintsSolver;
class Viewer;
class Timer;

class Simulator {
  public:
    Simulator();
    ~Simulator();

    bool init(int argc, char *argv[]);
    bool loop();

    Vector3s gravitational_constant;
    ScalarType dt;
    ScalarType t;
    bool terminated;
    bool enable_cuda;
    bool enable_viewer;
    bool save_mesh;
    unsigned int cuda_thread_per_block;

    std::string mesh_save_folder_path;
    std::unique_ptr<Timer> timer;

  private:
    void step();
    std::vector<std::unique_ptr<PhysicsBaseObject>> physics_objects;
    std::unique_ptr<Viewer> viewer;
    std::unique_ptr<ConstraintsSolver> constraints_solver;

    unsigned int time_step;
    unsigned int end_time_step;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif //!__SIMULATOR__