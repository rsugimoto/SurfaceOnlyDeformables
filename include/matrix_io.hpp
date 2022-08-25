#ifndef __MATRIX_IO_HPP__
#define __MATRIX_IO_HPP__

#include <fstream>
#include <iostream>
#include <string>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unsupported/Eigen/CXX11/Tensor>

namespace Eigen {
template <typename _Scalar, int _Rows, int _Cols, int _Options>
bool load_matrix(Eigen::Matrix<_Scalar, _Rows, _Cols, _Options> &mat, const std::string &filename) {
    std::ifstream in(filename.c_str(), std::ios::in | std::ios::binary);
    if (!in) {
        std::cerr << "load_matrix failed: " << filename << std::endl;
        return false;
    }

    Eigen::Index rows, cols;
    in.read((char *)&rows, sizeof(Eigen::Index));
    in.read((char *)&cols, sizeof(Eigen::Index));

    mat.resize(rows, cols);
    in.read((char *)mat.data(), sizeof(_Scalar) * rows * cols);

    in.close();

    if (!in) {
        std::cerr << "load_matrix failed: " << filename << std::endl;
        return false;
    }

    return true;
}

template <typename _Scalar, int _Rows, int _Cols, int _Options>
bool save_matrix(const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options> &mat, const std::string &filename) {
    std::ofstream out(filename.c_str(), std::ios::out | std::ios::binary);
    if (!out) {
        std::cerr << "save_matrix failed: " << filename << std::endl;
        return false;
    }

    Eigen::Index rows = mat.rows();
    Eigen::Index cols = mat.cols();
    out.write((const char *)&rows, sizeof(Eigen::Index));
    out.write((const char *)&cols, sizeof(Eigen::Index));

    out.write((const char *)mat.data(), sizeof(_Scalar) * mat.size());

    out.close();

    if (!out) {
        std::cerr << "save_matrix failed: " << filename << std::endl;
        return false;
    }

    return true;
}

template <typename _Scalar, int _Options, typename _StorageIndex>
bool load_matrix(Eigen::SparseMatrix<_Scalar, _Options, _StorageIndex> &mat, const std::string &filename) {
    std::ifstream in(filename.c_str(), std::ios::in | std::ios::binary);
    if (!in) {
        std::cerr << "load_matrix failed: " << filename << std::endl;
        return false;
    }

    Eigen::Index rows, cols, nonZeros, outerSize, innerSize;
    in.read((char *)&rows, sizeof(Eigen::Index));
    in.read((char *)&cols, sizeof(Eigen::Index));
    in.read((char *)&nonZeros, sizeof(Eigen::Index));
    in.read((char *)&outerSize, sizeof(Eigen::Index));
    in.read((char *)&innerSize, sizeof(Eigen::Index));

    mat.resize(rows, cols);
    mat.makeCompressed();
    mat.resizeNonZeros(nonZeros);

    in.read((char *)mat.valuePtr(), sizeof(_Scalar) * nonZeros);
    in.read((char *)mat.outerIndexPtr(), sizeof(_StorageIndex) * outerSize);
    in.read((char *)mat.innerIndexPtr(), sizeof(_StorageIndex) * nonZeros);

    mat.finalize();
    in.close();
    return true;
}

template <typename _Scalar, int _Options, typename _StorageIndex>
bool save_matrix(Eigen::SparseMatrix<_Scalar, _Options, _StorageIndex> &mat, const std::string &filename) {
    std::ofstream out(filename.c_str(), std::ios::out | std::ios::binary);
    if (!out) {
        std::cerr << "save_matrix failed: " << filename << std::endl;
        return false;
    }

    mat.makeCompressed();

    Eigen::Index rows = mat.rows();
    Eigen::Index cols = mat.cols();
    Eigen::Index nonZeros = mat.nonZeros();
    Eigen::Index outerSize = mat.outerSize();
    Eigen::Index innerSize = mat.innerSize();

    out.write((const char *)&rows, sizeof(Eigen::Index));
    out.write((const char *)&cols, sizeof(Eigen::Index));
    out.write((const char *)&nonZeros, sizeof(Eigen::Index));
    out.write((const char *)&outerSize, sizeof(Eigen::Index));
    out.write((const char *)&innerSize, sizeof(Eigen::Index));

    out.write((const char *)mat.valuePtr(), sizeof(_Scalar) * nonZeros);
    out.write((const char *)mat.outerIndexPtr(), sizeof(_StorageIndex) * outerSize);
    out.write((const char *)mat.innerIndexPtr(), sizeof(_StorageIndex) * nonZeros);

    out.close();
    return true;
}

template <typename _Scalar, int _Rank, int _Options>
bool load_tensor(Eigen::Tensor<_Scalar, _Rank, _Options> &tensor, const std::string &filename) {
    using TensorType = Eigen::Tensor<_Scalar, _Rank, _Options>;

    std::ifstream in(filename.c_str(), std::ios::in | std::ios::binary);
    if (!in) {
        std::cerr << "load_tensor failed: " << filename << std::endl;
        return false;
    }

    Eigen::array<Eigen::Index, _Rank> dims;
    for (typename TensorType::Index i = 0; i < _Rank; i++)
        in.read((char *)&dims[i], sizeof(typename TensorType::Index));

    tensor = TensorType(dims);
    in.read((char *)tensor.data(), sizeof(_Scalar) * tensor.size());

    in.close();
    return true;
}

template <typename _Scalar, int _Rank, int _Options>
bool save_tensor(const Eigen::Tensor<_Scalar, _Rank, _Options> &tensor, const std::string &filename) {
    using TensorType = Eigen::Tensor<_Scalar, _Rank, _Options>;

    std::ofstream out(filename.c_str(), std::ios::out | std::ios::binary);
    if (!out) {
        std::cerr << "save_tensor failed: " << filename << std::endl;
        return false;
    }

    for (int i = 0; i < tensor.NumDimensions; i++) {
        auto dim = tensor.dimension(i);
        out.write((const char *)&dim, sizeof(typename TensorType::Index));
    }

    out.write((const char *)tensor.data(), sizeof(_Scalar) * tensor.size());

    out.close();
    return true;
}

} // namespace Eigen

#endif //!__MATRIX_IO_HPP__
