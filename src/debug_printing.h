#include <iostream>
#include "csr.h"

/**
 * @brief This initialize the 5-point laplacian matrix (mn)x(mn) in CSR Format
 * 
 * @param A 
 * @param m 
 * @param n 
 */
void initializeCSRMatrix(CSRMatrix& A, int m, int n) {
    int matrix_size = m * n;
    int num_elements = m * n * m * n;

    // Initialize CSR matrix values and structure
    int index_curr, index_next, index_prev, index_up, index_down;
    int current_element = 0;

    for (int i = 0; i < matrix_size; ++i) {
        index_curr = i * matrix_size + i;

        // A_{i-1,j}
        index_up = index_curr - n;
        if (index_up >= i * matrix_size) {
            A.values[current_element] = -1.0;
            A.columns[current_element] = i - n;
            A.non_zeros++;
            current_element++;
        }

        // A_{i,j-1}
        index_prev = index_curr - 1;
        if (index_prev >= i * matrix_size) {
            A.values[current_element] = -1.0;
            A.columns[current_element] = i - 1;
            A.non_zeros++;
            current_element++;
        }

        // A_{i,j}
        A.values[current_element] = 4.0;
        A.columns[current_element] = i;
        A.non_zeros++;
        current_element++;

        // A_{i,j+1}
        index_next = index_curr + 1;
        if (index_next < (i + 1) * matrix_size) {
            A.values[current_element] = -1.0;
            A.columns[current_element] = i + 1;
            A.non_zeros++;
            current_element++;
        }
        
        // A_{i+1,j}
        index_down = index_curr + n;
        if (index_down < (i + 1) * matrix_size) {
            A.values[current_element] = -1.0;
            A.columns[current_element] = i + n;
            A.non_zeros++;
            current_element++;
        }

        // March the row pointer by 1
        A.row_ptr[i + 1] = current_element;
    }
}

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
