#ifndef __BARYCENTRIC_COORDINATES_CUH__
#define __BARYCENTRIC_COORDINATES_CUH__

#ifndef __device__
#define __device__
#endif // !__device__

#ifndef __host__
#define __host__
#endif // !__host__

#include "type_declaration.hpp"

// Copied from igl/barycentric_coordinates.cpp and __host__ __device__ qualifiers are added
template <typename DerivedP, typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedL>
__host__ __device__ void barycentric_coordinates(
    const Eigen::MatrixBase<DerivedP> &P, const Eigen::MatrixBase<DerivedA> &A, const Eigen::MatrixBase<DerivedB> &B,
    const Eigen::MatrixBase<DerivedC> &C, Eigen::PlainObjectBase<DerivedL> &L
) {
    using namespace Eigen;
    // http://gamedev.stackexchange.com/a/23745
    typedef Eigen::Array<typename DerivedP::Scalar, DerivedP::RowsAtCompileTime, DerivedP::ColsAtCompileTime> ArrayS;
    typedef Eigen::Array<typename DerivedP::Scalar, DerivedP::RowsAtCompileTime, 1> VectorS;

    const ArrayS v0 = B.array() - A.array();
    const ArrayS v1 = C.array() - A.array();
    const ArrayS v2 = P.array() - A.array();
    VectorS d00 = (v0 * v0).rowwise().sum();
    VectorS d01 = (v0 * v1).rowwise().sum();
    VectorS d11 = (v1 * v1).rowwise().sum();
    VectorS d20 = (v2 * v0).rowwise().sum();
    VectorS d21 = (v2 * v1).rowwise().sum();
    VectorS denom = d00 * d11 - d01 * d01;
    L.resize(P.rows(), 3);
    L.col(1) = (d11 * d20 - d01 * d21) / denom;
    L.col(2) = (d00 * d21 - d01 * d20) / denom;
    L.col(0) = (ScalarType)1.0 - (L.col(1) + L.col(2)).array();
}

#endif //__BARYCENTRIC_COORDINATES_CUH__