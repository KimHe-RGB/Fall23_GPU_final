#include <iostream>
#include "csr.h"

/**
 * @brief Print out a CSRMatrix as a full matrix
 * 
 * @param A 
 */
void print_csr_matrix(const CSRMatrix& A) {
    int rows = A.rows;

    for (int i = 0; i < rows; ++i) {
        int valueIndex = A.row_ptr[i];
        for (int j = 0; j < rows; ++j) {
            if (valueIndex < A.row_ptr[i + 1] && A.columns[valueIndex] == j) {
                std::cout << A.values[valueIndex] << " ";
                ++valueIndex;
            } else {
                std::cout << " 0 ";
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

/**
 * @brief print a diagonal matrix
 * 
 * @param D 
 * @param dim 
 */
void print_diagonal(const double* D, const int dim){
    for (size_t i = 0; i < dim; i++)
    {
        std::cout << D[i] << std::endl;
    }
    std::cout << std::endl;
}

/**
 * @brief Print out a CSRMatrix's values, rows, and columns
 * 
 * @param A 
 */
void print_csr_matrix_info(const CSRMatrix& A) {
    // Print values
    std::cout << "Values: ";
    for (int i = 0; i < A.non_zeros; ++i) {
        std::cout << A.values[i] << " ";
    }
    std::cout << std::endl;

    // Print row pointers
    std::cout << "Row Pointers: ";
    for (int i = 0; i <= A.rows; ++i) {
        std::cout << A.row_ptr[i] << " ";
    }
    std::cout << std::endl;

    // Print columns
    std::cout << "Columns: ";
    for (int i = 0; i < A.non_zeros; ++i) {
        std::cout << A.columns[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "non-zeros: " << A.non_zeros << std::endl;
    std::cout << "capacity: " << A.capacity << std::endl;
}
