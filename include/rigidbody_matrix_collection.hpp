#ifndef __RIGIDBODY_MATRIX_COLLECTION__
#define __RIGIDBODY_MATRIX_COLLECTION__

#include "physics_base_matrix_collection.hpp"

class RigidbodyMatrixCollection : public PhysicsBaseMatrixCollection {
  public:
    RigidbodyMatrixCollection();
    ~RigidbodyMatrixCollection();

    bool init(const nlohmann::json &config) override;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    friend class RigidbodyObject;
};

#endif //!__RIGIDBODY_MATRIX_COLLECTION__