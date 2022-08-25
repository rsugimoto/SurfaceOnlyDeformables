#include "rigidbody_matrix_collection.hpp"

#include <iostream>

RigidbodyMatrixCollection::RigidbodyMatrixCollection() : PhysicsBaseMatrixCollection() {}
RigidbodyMatrixCollection::~RigidbodyMatrixCollection() {}

bool RigidbodyMatrixCollection::init(const nlohmann::json &config) {
    if (!PhysicsBaseMatrixCollection::init(config)) return false;
    std::cout << "type: Rigidbody" << std::endl;
    return true;
}
