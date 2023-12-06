#include <iostream>
#include "debug_printing.h"

/**
 * @brief Decompose spd matrix A into LDLt, such that A = L * D * L^T
 * 
 * @param A spd matrix in CSR Format
 * @param L Lower Triangular matrix in CSR Format
 * @param D diagonal matrix stored as a vector
 */
void ldlt_cholesky_decomposition_seq(CSRMatrix& A, CSRMatrix& L, CSRMatrix& Lt, double* D) {

    // Initialize L and Lt matrix
    L.rows = A.rows;
    L.non_zeros = 0;
    L.row_ptr[0] = 0;
    Lt.rows = A.rows;
    Lt.non_zeros = 0;
    Lt.row_ptr[0] = 0;

    // LDL Cholesky Decomposition    
    for (int j = 0; j < A.rows; j++) {
        double sumL = 0.0;
        // Calculate the diagonal element of L and update D
        for (int k = 0; k < j; k++) {
            sumL += L.get_ij(j, k) * L.get_ij(j, k) * D[k];
        }
        D[j] = A.get_ij(j,j) - sumL;  
        L.set_ij(j,j,1);
        Lt.set_ij(j,j,1);
        // std::cout << j << " : " << D[j] << " - " << sumL << std::endl;

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
                Lt.set_ij(j,i,Lij);
            }
        }
    }

    // std::cout << "Values of L: ";
    // for (int i = 0; i < L.non_zeros; ++i) {
    //     std::cout << L.values[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "Values of Lt: ";
    // for (int i = 0; i < Lt.non_zeros; ++i) {
    //     std::cout << Lt.values[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << std::endl;
}


/**
 * @brief forward substitute to solve x = b \ L, L lower triangular in CSR format
 * @param L Lower Triangular matrix in CSR Format
 * @param b vector 
 * @param x result vector
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
 * @brief backward substitute to solve x = b \ Lt
 * @param U Upper Triangular matrix in CSR Format
 * @param b vector 
 * @param x result vector
 */
void backward_substitute(const double* b, const CSRMatrix& U, double* x) {    
    int rows = U.rows;
    // Backward substitution
    for (int i = rows - 1; i >= 0; i--) {
        // Compute the sum of U(i, j) * x(j) for j = i+1 to rows-1
        double sum = 0.0;
        for (int k = U.row_ptr[i] + 1; k < U.row_ptr[i + 1]; ++k) {
            int j = U.columns[k];
            sum += U.values[k] * x[j];
        }
        // Solve for x(i)
        x[i] = (b[i] - sum) / U.values[U.row_ptr[i]];
    }
}


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
        x[i] = x[i] / D[i];
    }
}

/**
 * @brief This initialize the 5-point laplacian matrix (mn)x(mn) in CSR Format
 * 
 * @param A 
 * @param m 
 * @param n 
 */
void initializeCSRMatrix(CSRMatrix& A, int m, int n) {
    int matrix_size = m * n;

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
