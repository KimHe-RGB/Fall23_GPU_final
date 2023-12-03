#include <iostream>
#include "debug_printing.h"

/**
 * @brief Decompose spd matrix A into LDLt, such that A = L * D * L^T
 * 
 * @param A spd matrix in CSR Format
 * @param L Lower Triangular matrix in CSR Format
 * @param D diagonal matrix stored as a vector
 */
void ldlt_cholesky_decomposition_seq(CSRMatrix& A, CSRMatrix& L, double* D) {

    // Initialize L matrix
    L.rows = A.rows;
    L.non_zeros = 0;
    L.row_ptr[0] = 0;

    // LDL Cholesky Decomposition    
    for (int j = 0; j < A.rows; ++j) {
        double sumL = 0.0;
        // Calculate the diagonal element of L and update D
        for (int k = 0; k < j; k++) {
            sumL += L.get_ij(j, k) * L.get_ij(j, k) * D[k];
        }
        D[j] = A.get_ij(j,j) - sumL;  
        L.set_ij(j,j,1);

        // Calculate the off-diagonal elements of L on Row i, for i > j
        for (int i = j+1; i < L.rows; i++) {
            double sumL2 = 0.0; 

            double Aij = A.get_ij(i, j);
            // sum up previous L values, find whether Lik or Ljk is zero or non-zero
            for (int k = 0; k < j; k++)
            {
                // \sum_{k=1}^{j-1} L_{ik}*L_{jk}*D_{k}
                sumL2 += L.get_ij(i,k) * L.get_ij(j,k) * D[k];
            }
            // store Lij if it is non-zeros, assume D[j] != 0
            if (Aij - sumL2 != 0) {
                double Lij = (Aij - sumL2) / D[j];
                L.set_ij(i,j,Lij); 
            }
        }
    }
}


/**
 * @brief forward substitute to solve x = b \ L, L lower triangular in CSR format
 * @param L Lower Triangular matrix in CSR Format
 * @param b vector 
 * @param x vector
 */
void forward_substitute(const double* b, const CSRMatrix& L, double* x) {
    int rows = L.rows;

    // Forward substitution
    for (int i = 0; i < rows; ++i) {
        // Compute the sum of L(i, j) * x(j) for j = 0 to i-1
        double sum = 0.0;
        for (int k = L.row_ptr[i]; k < L.row_ptr[i + 1]; ++k) {
            int j = L.columns[k];
            sum += L.values[k] * x[j];
        }

        // Solve for x(i)
        x[i] = (b[i] - sum) / L.values[L.row_ptr[i + 1] - 1];
    }
}

/**
 * @brief backward substitute to solve x = b \ Lt, L lower triangular in CSR format, and Lt is its transpose which is not given
 * @param L Lower Triangular matrix in CSR Format
 * @param b vector 
 * @param x vector
 */
void backward_substitute(const double* y, const CSRMatrix& U, double* x) {
    int rows = U.rows;

    // Backward substitution
    for (int i = rows - 1; i >= 0; --i) {
        // Compute the sum of U(i, j) * x(j) for j = i+1 to rows-1
        double sum = 0.0;
        for (int k = U.row_ptr[i] + 1; k < U.row_ptr[i + 1]; ++k) {
            int j = U.columns[k];
            sum += U.values[k] * x[j];
        }

        // Solve for x(i)
        x[i] = (y[i] - sum) / U.values[U.row_ptr[i]];
    }
}
// void backward_substitute(const double* y, const CSRMatrix& L, double* x) {
//     int rows = L.rows;

//     // Backward substitution
//     for (int i = rows - 1; i >= 0; --i) {
//         // Compute the sum of L(i, j) * x(j) for j = i+1 to rows-1
//         double sum = 0.0;
//         for (int k = L.row_ptr[i] + 1; k < L.row_ptr[i + 1]; ++k) {
//             int j = L.columns[k];
//             sum += L.values[k] * x[j];
//         }

//         // Solve for x(i)
//         x[i] = (y[i] - sum) / L.values[L.row_ptr[i]];
//     }
// }

/**
 * @brief perform an elementwise division x ./ D
 * 
 * @param D diagonal matrix D stored as a vector
 * @param x result vector
 */
void elementwise_division_vector(double* D, double* x, int dim)
{
    for (size_t i = 0; i < dim; i++)
    {
        x[i] = x[i] - D[i];
    }
    
}


void transposeCSR(const CSRMatrix& input, CSRMatrix& output) {
    // Swap rows and columns
    output.rows = input.rows;
    output.capacity = input.capacity;
    output.non_zeros = input.non_zeros;

    // Allocate memory for the transposed matrix
    output.values = new double[output.capacity];
    output.columns = new int[output.capacity];
    output.row_ptr = new int[output.rows + 1];

    // Count the number of non-zeros in each column
    int* columnCounts = new int[input.rows];
    std::fill(columnCounts, columnCounts + input.rows, 0);

    for (int i = 0; i < input.rows; ++i) {
        for (int j = input.row_ptr[i]; j < input.row_ptr[i + 1]; ++j) {
            columnCounts[input.columns[j]]++;
        }
    }

    // Fill row_ptr for the transposed matrix
    output.row_ptr[0] = 0;
    for (int i = 1; i <= output.rows; ++i) {
        output.row_ptr[i] = output.row_ptr[i - 1] + columnCounts[i - 1];
        columnCounts[i - 1] = 0;  // Reset for the next iteration
    }

    // Fill values and columns for the transposed matrix
    for (int i = 0; i < input.rows; ++i) {
        for (int j = input.row_ptr[i]; j < input.row_ptr[i + 1]; ++j) {
            int col = input.columns[j];
            int index = output.row_ptr[col] + columnCounts[col];
            output.values[index] = input.values[j];
            output.columns[index] = i;
            columnCounts[col]++;
        }
    }

    // Clean up temporary memory
    delete[] columnCounts;
}