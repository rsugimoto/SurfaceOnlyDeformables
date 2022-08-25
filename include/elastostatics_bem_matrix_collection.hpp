#ifndef __ELASTOSTATICS_BEM_MATRIX_COLLECTION_HPP__
#define __ELASTOSTATICS_BEM_MATRIX_COLLECTION_HPP__

#include "elastics_bem_matrix_collection.hpp"
#include <deque>

class ElastostaticsBEMMatrixCollection : public ElasticsBEMMatrixCollection {
  public:
    ElastostaticsBEMMatrixCollection();
    ~ElastostaticsBEMMatrixCollection();

    bool init(const nlohmann::json &config) override;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    friend class ElastostaticsBEMObject;
};

#endif //!__ELASTOSTATICS_BEM_MATRIX_COLLECTION_HPP__