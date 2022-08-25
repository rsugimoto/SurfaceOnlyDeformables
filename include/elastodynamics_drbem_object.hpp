#ifndef __ELASTODYNAMICS_DRBEM_OBJECT_HPP__
#define __ELASTODYNAMICS_DRBEM_OBJECT_HPP__

#include "physics_base_object.hpp"
#include <deque>

class ElastodynamicsDRBEMMatrixCollection;

class ElastodynamicsDRBEMObject : public PhysicsBaseObject {
  public:
    ElastodynamicsDRBEMObject(
        const std::shared_ptr<const ElastodynamicsDRBEMMatrixCollection> matrix_collection, ScalarType dt,
        const Vector3s &gravitational_constant
    );
    ~ElastodynamicsDRBEMObject();

    bool init(const nlohmann::json &config) override;

    void estimate_next_state() override;
    void confirm_next_state() override;

  protected:
    void update_V_estimate_matrices() override;

  private:
    VectorXs b;
    VectorXs u_unconstrained;
    std::deque<VectorXs> u_hist;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif //!__ELASTODYNAMICS_DRBEM_OBJECT_HPP__