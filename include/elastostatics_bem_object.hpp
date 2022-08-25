#ifndef __ELASTOSTATICS_BEM_OBJECT_HPP__
#define __ELASTOSTATICS_BEM_OBJECT_HPP__

#include "physics_base_object.hpp"

class ElastostaticsBEMMatrixCollection;

class ElastostaticsBEMObject : public PhysicsBaseObject {
  public:
    ElastostaticsBEMObject(
        const std::shared_ptr<const ElastostaticsBEMMatrixCollection> matrix_collection, ScalarType dt,
        const Vector3s &gravitational_constant
    );
    ~ElastostaticsBEMObject();

    bool init(const nlohmann::json &config) override;

    void estimate_next_state() override;
    void confirm_next_state() override;

  protected:
    void update_V_estimate_matrices() override;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif //!__ELASTOSTATICS_BEM_OBJECT_HPP__