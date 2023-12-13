#ifndef __CSR_H
#include "csr.h"
#endif

#ifndef __DEBUG_PRINT_H
#define __DEBUG_PRINT_H
#include <iostream>
#include <iomanip>
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
                std::cout << std::setprecision(5) << " " << A.values[valueIndex] << " ";
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
    for (int i = 0; i < A.row_ptr[A.rows]; ++i) {
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
    for (int i = 0; i < A.row_ptr[A.rows]; ++i) {
        std::cout << A.columns[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "non-zeros: " << A.row_ptr[A.rows] << std::endl;
    std::cout << "capacity: " << A.capacity << std::endl;
}


/**
 * @brief print out 2d heat map
 * 
 * @param heat 
 * @param m 
 * @param n 
 */
void print_heat_map(double* heat, int m, int n) {
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cout << heat[i*n + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


/**
 * @brief print out CSR from Arrays
 * 
 * @param L_values 
 * @param L_columns 
 * @param L_row_ptr 
 * @param M 
 * @param N 
 */
void printCSR(double* L_values, int* L_columns, int* L_row_ptr, int M, int N){
    for (int i = 0; i < M*N; ++i) {
        int valueIndex = L_row_ptr[i];
        for (int j = 0; j < M*N; ++j) {
            if (valueIndex < L_row_ptr[i + 1] && L_columns[valueIndex] == j) {
                std::cout << L_values[valueIndex] << " ";
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
 * @brief Print out the Array 
 * 
 * @param D 
 * @param dim 
 */
void printArray(double *D, int dim) {
    for (int i = 0; i < dim; i++)
    {
        std::cout << D[i] << std::endl;
    }
    
}

#endif