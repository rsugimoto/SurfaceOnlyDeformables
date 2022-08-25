#ifndef __ELASTODYNAMICS_CQBEM_OBJECT_HPP__
#define __ELASTODYNAMICS_CQBEM_OBJECT_HPP__

#include "physics_base_object.hpp"
#include "wavelets.hpp"
#include <deque>

class ElastodynamicsCQBEMMatrixCollection;

class ElastodynamicsCQBEMObject : public PhysicsBaseObject {
  public:
    ElastodynamicsCQBEMObject(
        const std::shared_ptr<const ElastodynamicsCQBEMMatrixCollection> matrix_collection, ScalarType dt,
        const Vector3s &gravitational_constant
    );
    ~ElastodynamicsCQBEMObject();

    bool init(const nlohmann::json &config) override;

    void estimate_next_state() override;
    void confirm_next_state() override;

  protected:
    void update_V_estimate_matrices() override;

  private:
    int instance_id;

    std::deque<Vector6s> hist_acc_vectors;
    std::deque<CompressedVector<>> hist_u_compressed_vectors, hist_p_compressed_vectors;
    std::deque<VectorXs> hist_u_vectors, hist_p_vectors;

    VectorXs b;
    VectorXs u_unconstrained;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif //!__ELASTODYNAMICS_CQBEM_OBJECT_HPP__