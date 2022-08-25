#ifndef __TYPE_DECLARATION_HPP__
#define __TYPE_DECLARATION_HPP__

// #define EIGEN_DONT_PARALLELIZE // Don't use OpenMP for Eigen
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int64_t

#if MKL_AVAILABLE
#define EIGEN_USE_MKL_ALL
#endif

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <complex>
#include <unsupported/Eigen/CXX11/Tensor>

using ScalarType = double;
using IntType = int;
constexpr auto StorageOrder = Eigen::RowMajor; // Need to change a line in B_angular manually

using SparseMatrix = Eigen::SparseMatrix<ScalarType, Eigen::RowMajor>;
using DiagonalMatrixXs = Eigen::DiagonalMatrix<ScalarType, Eigen::Dynamic>;
using PermutationMatrix = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>;

using Vector3s = Eigen::Matrix<ScalarType, 3, 1>;
using RowVector3s = Eigen::Matrix<ScalarType, 1, 3>;
using Vector6s = Eigen::Matrix<ScalarType, 6, 1>;
using VectorXs = Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>;
using Matrix3s = Eigen::Matrix<ScalarType, 3, 3, StorageOrder>;
using MatrixX3s = Eigen::Matrix<ScalarType, Eigen::Dynamic, 3, StorageOrder>;
using MatrixX6s = Eigen::Matrix<ScalarType, Eigen::Dynamic, 6, StorageOrder>;
using Matrix3Xs = Eigen::Matrix<ScalarType, 3, Eigen::Dynamic, StorageOrder>;
using MatrixXs = Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic, StorageOrder>;
using TensorXs = Eigen::Tensor<ScalarType, 3, StorageOrder>;

using Vector3i = Eigen::Matrix<IntType, 3, 1>;
using RowVector3i = Eigen::Matrix<IntType, 1, 3>;
using VectorXi = Eigen::Matrix<IntType, Eigen::Dynamic, 1>;
using MatrixX3i = Eigen::Matrix<IntType, Eigen::Dynamic, 3, StorageOrder>;
using MatrixX2i = Eigen::Matrix<IntType, Eigen::Dynamic, 2, StorageOrder>;

using ComplexType = std::complex<ScalarType>;
using Vector3c = Eigen::Matrix<ComplexType, 3, 1>;
using RowVector3c = Eigen::Matrix<ComplexType, 1, 3>;
using VectorXc = Eigen::Matrix<ComplexType, Eigen::Dynamic, 1>;
using Matrix3c = Eigen::Matrix<ComplexType, 3, 3, StorageOrder>;
using MatrixX3c = Eigen::Matrix<ComplexType, Eigen::Dynamic, 3, StorageOrder>;
using MatrixXc = Eigen::Matrix<ComplexType, Eigen::Dynamic, Eigen::Dynamic, StorageOrder>;
using TensorXc = Eigen::Tensor<ComplexType, 3, StorageOrder>;

using MatrixXb = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, StorageOrder>;

#endif //!__TYPE_DECLARATION_HPP__