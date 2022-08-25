#ifndef __CROSS_PRODUCT_MATRIX_HPP__
#define __CROSS_PRODUCT_MATRIX_HPP__

#include "type_declaration.hpp"

template <class T> Matrix3s cross_product_matrix(const T &vec) {
    Matrix3s mat;
    mat << 0.0, -vec(2), vec(1), vec(2), 0.0, -vec(0), -vec(1), vec(0), 0.0;
    return mat;
};

#endif //__CROSS_PRODUCT_MATRIX_HPP__