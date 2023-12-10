#include <iostream>

#ifndef __CSR_H
#include "csr.h"
#endif

#ifndef __GLOBAL_H
#include "global.h"
#endif

#ifndef __DEBUG_PRINT_H
#include "debug_printing.h"
#endif

#ifndef __LIN_ALG_H
#define __LIN_ALG_H
/**
 * @brief Decompose spd matrix A into LDLt, such that A = L * D * L^T
 * 
 * @param A spd matrix in CSR Format
 * @param L Lower Triangular matrix in CSR Format
 * @param D diagonal matrix stored as a vector
 */
void ldlt_cholesky_decomposition_seq(CSRMatrix& A, CSRMatrix& L, CSRMatrix& Lt, double* D, int m, int n) {

    // Initialize L and Lt matrix 
    // to avoid too many shifting, we pre-allocate the non-zero terms with zero
    L.rows = A.rows;
    L.non_zeros = 0;
    Lt.rows = A.rows;
    Lt.non_zeros = 0;

    // L: row 0 only have 1 element
    L.row_ptr[0] = 0; int count = 0;
    // L: row i from 1 to n has i non-zero elements
    for (int i = 1; i < n+1; i++)
    {
        L.row_ptr[i] = L.row_ptr[i-1] + i; 
        for (int j = 0; j < i; j++)
        {
            L.columns[count] = j; L.values[count] = 0; count++;
        }
    }
    // L: row i from n+1 to m*n has n + 1 non-zero elements
    for (int i = n+1; i < m*n+1; i++)
    {
        L.row_ptr[i] = L.row_ptr[i-1] + n + 1; 
        for (int j = i - n - 1; j < i; j++)
        {
            L.columns[count] = j; L.values[count] = 0; count++;
        }
    }
    L.non_zeros = count;
    
    // Lt: row i from 0 to m-n have n + 1 non-zero elements
    count = 0; L.row_ptr[0] = 0;
    for (int i = 1; i < m*n-n+1; i++)
    {
        Lt.row_ptr[i] = Lt.row_ptr[i-1] + n + 1; 
        for (int j = -1; j < n; j++)
        {
            Lt.columns[count] = i+j; Lt.values[count] = 0; count++;
        }
    }
    // Lt: row i from m-n to m have m-i non-zero element
    for (int i = m*n-n; i < m*n; i++)
    {
        Lt.row_ptr[i+1] = Lt.row_ptr[i] + m*n - i; 
        for (int j = i; j < m*n; j++)
        {
            Lt.columns[count] = j; Lt.values[count] = 0; count++;
        }
    }
    Lt.non_zeros = count;

    std::cout << "total rows: " << A.rows << std::endl;
    // LDL Cholesky Decomposition: we know L non-zero term position now   
    for (int j = 0; j < A.rows; j++) {
        double sumL = 0.0;
        // Calculate the diagonal element of L and update D
        for (int k = 0; k < j; k++) {
            double Ljk = L.get_ij(j, k);
            sumL += Ljk * Ljk * D[k];
        }
        D[j] = A.get_ij(j,j) - sumL;  
        L.set_ij(j,j,1);
        Lt.set_ij(j,j,1);
        // std::cout << j << " : " << D[j] << " - " << sumL << std::endl;

        // Calculate the off-diagonal elements of L on Row i, for all i > j, there are at most n+1 rows to update
        for (int i = j+1; i < j+n+1; i++) {
            double sumL2 = 0.0; 

            double Aij = A.get_ij(i, j);
            // sum up previous L values, find whether Lik or Ljk is zero or non-zero, there are at most n+1 non zero elements
            // TBD: unroll first n loops
            int st = std::max(0, j-n);
            for (int k = st; k < j; k++)
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
        // finished row
        // std::cout << "finished col: " << j << std::endl;
    }
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
 * @brief Given A = LDL', solve LDL'x = b
 * 
 */
void solveAxb(CSRMatrix &L, CSRMatrix &Lt, double *D, double *b, double *x, const int MATRIX_DIM)
{
    double *x_temp = (double *) malloc(MATRIX_DIM*sizeof(double));
    forward_substitute(b, L, x_temp);
    elementwise_division_vector(D, x_temp, MATRIX_DIM);
    backward_substitute(x_temp, Lt, x);
    free(x_temp);
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


/**
 * @brief This initialize the 5-point laplacian matrix in CSR Format for Backward Euler method, I+ht*invhsq*A
 * 
 * @param A 
 * @param m dimension X
 * @param n dimension Y
 * @param htinvhsq ht * invhsq
 */
void initBackwardEulerCSRMatrix(CSRMatrix& A, double htinvhsq, int m, int n) {
    int matrix_size = m * n;

    // Initialize CSR matrix values and structure
    int index_curr, index_next, index_prev, index_up, index_down;
    int current_element = 0;

    for (int i = 0; i < matrix_size; ++i) {
        index_curr = i * matrix_size + i;

        // A_{i-1,j}
        index_up = index_curr - n;
        if (index_up >= i * matrix_size) {
            A.values[current_element] = -htinvhsq;
            A.columns[current_element] = i - n;
            A.non_zeros++;
            current_element++;
        }

        // A_{i,j-1}
        index_prev = index_curr - 1;
        if (index_prev >= i * matrix_size && i%n != 0) {
            A.values[current_element] = -htinvhsq;
            A.columns[current_element] = i - 1;
            A.non_zeros++;
            current_element++;
        }

        // A_{i,j}
        A.values[current_element] = 1 + 4.0*htinvhsq;
        A.columns[current_element] = i;
        A.non_zeros++;
        current_element++;

        // A_{i,j+1}
        index_next = index_curr + 1;
        if (index_next < (i + 1) * matrix_size && (i+1)%n != 0) {
            A.values[current_element] = -htinvhsq;
            A.columns[current_element] = i + 1;
            A.non_zeros++;
            current_element++;
        }
        
        // A_{i+1,j}
        index_down = index_curr + n;
        if (index_down < (i + 1) * matrix_size) {
            A.values[current_element] = -htinvhsq;
            A.columns[current_element] = i + n;
            A.non_zeros++;
            current_element++;
        }

        // March the row pointer by 1
        A.row_ptr[i + 1] = current_element;
    }
}


/**
 * @brief Use Backward Euler method to solve heat equation
 * For each time step, we solve the linear system u_{k+1} = (I + ht*invhsq * A) \ (u_k + ht*invhsq * f);
 * 
 * @param f boudary correction term
 * @param u the heat map being updated
 * @param D space for the diagonal matrix
 * @param m 
 * @param n 
 */
void Backward_Euler_CSR(double *f, double *u, double* D, const int m, const int n)
{
    // u_{k+1} = (I + ht*invhsq*A) \ (u_k + ht*invhsq*f);
    double t = 0.0;
    const int MATRIX_DIM = m*n;

    computeBoundaryCondition(f, u, m, n);
    // left side
    std::cout << "init kernel" << std::endl;
    CSRMatrix A = CSRMatrix(MATRIX_DIM, 5*m*n);
    initBackwardEulerCSRMatrix(A, tau*invhsq, m, n); // initialize I + ht*invhsq*A
    // print_csr_matrix(A);
    CSRMatrix L = CSRMatrix(MATRIX_DIM, 5*m*m*n);
    CSRMatrix Lt = CSRMatrix(MATRIX_DIM, 5*m*m*n);
    std::cout << "start cholesky" << std::endl;
    ldlt_cholesky_decomposition_seq(A, L, Lt, D, m, n);
    // print_diagonal(D, MATRIX_DIM);
    // right side
    double *b = (double *) malloc(MATRIX_DIM*MATRIX_DIM*sizeof(double));
    std::cout << "start running euler steps" << std::endl;
    
    int total_steps = endT / tau;

    for (int p = 0; p < total_steps; p++)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                b[i*n+j] = u[i*n+j] + tau*invhsq*f[i*n+j];
            }
        }
        // std::cout << "b: \n";
        // print_heat_map(b, m, n);
        solveAxb(L, Lt, D, b, u, MATRIX_DIM);
        // std::cout << "u: \n";
        // print_heat_map(u, m, n);
    }
    free(b);
}



#endif