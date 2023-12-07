/**
 * @file csr.h
 * @author Xujin He (you@domain.com)
 * @brief This header file defines CSR struct for storing the sparse matrix
 * @version 0.1
 * @date 2023-11-29
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#ifndef __CSR_H
#define __CSR_H
#include <iostream>

// i-th row length: rowInd[i+1] - rowInd[i]
// i-th row access: for (i = rowInd[i]; i < rowInd[i+1]; i++) V[i]
// access element ij: V[rowInd[i] + j]; j must leq rowInd[i+1]-rowInd[i]
// it is hard to visit column wise

// CSR Storage
struct CSRMatrix {
    double* values;   // Array storing non-zero values of the matrix
    int* columns;     // Array storing column indices of non-zero values
    int* row_ptr;      // Array storing row pointers (indices where each row starts)
    int rows;          // Number of rows in the matrix
    int capacity;      // maximum number of non-zeros
    int non_zeros;     // Number of non-zero elements in the matrix; Essentially "Capacity";

    // Constructor to initialize CSR Matrix
    CSRMatrix(int rows, int capacity) : rows(rows), capacity(capacity) {
        values = new double[capacity];
        columns = new int[capacity];
        row_ptr = new int[rows + 1];
        non_zeros = 0;
    }

    // Destructor to free allocated memory
    ~CSRMatrix() {
        delete[] values;
        delete[] columns;
        delete[] row_ptr;
    }

    double get_ij(int i, int j) const 
    {
        int start = row_ptr[i];
        int end = row_ptr[i + 1];
        for (int k = start; k < end; k++) {
            if (columns[k] == j) {
                return values[k];
            }
        }
        // Column not found, as columns are in ascending order
        return 0;
    }

    void set_ij(int i, int j, double value) 
    {
        int start = row_ptr[i];
        int end = row_ptr[i + 1];
        // Check if the element already exists in the matrix
        for (int k = start; k < end; k++) {
            if (columns[k] == j) {
                values[k] = value;  // Update existing value
                return;
            }
        }
        return;

        // // for large matrix, this branch is the most time-consuming step so we need to avoid, 
        // // we initialize L so that this branch is avoided
        //
        // // If the element doesn't exist, add it to the matrix
        // std::cout << "Bad Branch: undesired insertion at " << i << ", " << j << std::endl;
        // int insertPos = end;
        // for (int k = start; k < end; k++) {
        //     if (columns[k] > j) {
        //         insertPos = k;
        //         break;
        //     }
        // }

        // // Shift all existing non_zeros values and columns to make space for the new element
        // for (int k = non_zeros; k > insertPos; k--) {
        //     values[k] = values[k - 1];
        //     columns[k] = columns[k - 1];
        // }

        // // Insert the new element
        // values[insertPos] = value;
        // columns[insertPos] = j;

        // // Update row pointers after insertion
        // for (int k = i + 1; k <= rows; k++) {
        //     row_ptr[k]++;
        // }
        // non_zeros++;

        // std::cout << i << ", " << j << " = " << values[insertPos] << std::endl;
        // std::cout << "Values: ";
        // for (int i = 0; i < non_zeros; ++i) {
        //     std::cout << values[i] << " ";
        // }
        // std::cout << std::endl;
    }
};

// CSC Storage
struct CSCMatrix {
    double* values;     // Array storing non-zero values of the matrix
    int* rows;          // Array storing row indices of non-zero values
    int* col_ptr;       // Array storing column pointers (indices where each column starts)
    int cols;           // Number of columns in the matrix
    int capacity;      // maximum number of non-zeros
    int non_zeros;      // Number of non-zero elements in the matrix

    // Constructor to initialize CSCMatrix
    CSCMatrix(int cols, int capacity) : cols(cols), non_zeros(capacity) {
        values = new double[capacity];
        rows = new int[capacity];
        col_ptr = new int[cols + 1];
        non_zeros = 0;
    }

    // Destructor to free allocated memory
    ~CSCMatrix() {
        delete[] values;
        delete[] rows;
        delete[] col_ptr;
    }
};

#endif