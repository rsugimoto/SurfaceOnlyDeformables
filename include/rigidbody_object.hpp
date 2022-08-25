#ifndef __RIGIDBODY_OBJECT__
#define __RIGIDBODY_OBJECT__

#include "physics_base_object.hpp"

class RigidbodyMatrixCollection;

class RigidbodyObject : public PhysicsBaseObject {
  public:
    RigidbodyObject(
        const std::shared_ptr<const RigidbodyMatrixCollection> matrix_collection, ScalarType dt,
        const Vector3s &gravitational_constant
    );
    ~RigidbodyObject();

    bool init(const nlohmann::json &config) override;

    void estimate_next_state() override;
    void confirm_next_state() override;

  protected:
    void update_V_estimate_matrices() override;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif //!__RIGIDBODY_OBJECT__