#include <iostream>

#ifndef __GLOBAL_H
#include "global.h"
#endif

#ifndef __DEBUG_PRINT_H
#include "debug_printing.h"
#endif

#define index(i, j, N)  ((i)*(N)) + (j)


void ldlt_cholesky_decomposition_seq_array(double* A, double* L, double* D, int m, int n) {
    const int dim = m*n;
    // LDL Cholesky Decomposition    
    for (int j = 0; j < dim; j++) {
        double sumL = 0.0;
        // Calculate the diagonal element of L and update D
        for (int k = 0; k < j; k++) {
            double Ljk = L[index(j, k, dim)];
            sumL += Ljk * Ljk * D[k];
        }
        D[j] = A[index(j,j,dim)] - sumL; 
        L[index(j,j,dim)] = 1; 
        // std::cout << j << " : " << D[j] << " - " << sumL << std::endl;

        // Calculate the off-diagonal elements of L on Row i, for i > j
        for (int i = j+1; i < dim; i++) {
            double sumL2 = 0.0; 

            double Aij = A[index(i,j,dim)];
            // sum up previous L values, find whether Lik or Ljk is zero or non-zero
            for (int k = 0; k < j; k++)
            {
                // \sum_{k=1}^{j-1} L_{ik}*L_{jk}*D_{k}
                sumL2 += L[index(i,k,dim)] * L[index(j,k,dim)] * D[k];
            }
            // store Lij if it is non-zeros, assume D[j] != 0
            if (Aij - sumL2 != 0) {
                double Lij = (Aij - sumL2) / D[j];
                L[index(i,j,dim)] = Lij; 
            }
        }
    }
}

/**
 * @brief forward substitute to solve x = b \ L, L lower triangular
 * @param L Lower Triangular square matrix with side length @param m * @param n
 * @param b vector 
 * @param x result vector
 */
void forward_substitute_seq_array(double* b, double* L, double* x, int m, int n) {
    const int dim = m*n;
    for (int i = 0; i < dim; ++i) {
        x[i] = b[i];
        for (int j = 0; j < i; ++j) {
            x[i] -= L[index(i, j, dim)] * x[j];
        }
        x[i] /= L[index(i, i, dim)];
    }
}

/**
 * @brief backward substitute to solve x = b \ Lt
 * @param L Lower Triangular square matrix with side length @param m * @param n
 * @param b vector 
 * @param x result vector
 */
void backward_substitute_seq_array(double* b, double* L, double* x, int m, int n) { 
    const int dim = m*n;   
    for (int i = dim - 1; i >= 0; --i) {
        x[i] = b[i];
        for (int j = i + 1; j < dim; ++j) {
            x[i] -= L[index(j, i, dim)] * x[j];
        }
        x[i] /= L[index(i, i, dim)];
    }
}

/**
 * @brief init the matrix for linear system being solved in Backward Euler
 * 
 * @param A (I + ht*invhsq*A)
 * @param htinvhsq 1/h^2 * tau, where h is grid size and tau is time step size
 * @param m 
 * @param n 
 */
void initBackwardEulerMatrix_array(double* A, double htinvhsq, int m, int n) {
    int matrix_size = m * n;

    int index_curr, index_next, index_prev, index_up, index_down;
    for (int i = 0; i < matrix_size; i++) {
        index_curr = i * matrix_size + i;
        // A_{i-1,j}
        std::cout << index_curr << std::endl;
        index_up = index_curr - n;
        if (index_up >= i * matrix_size) {
            A[index_up] = -htinvhsq;
        }
        // A_{i,j-1}
        index_prev = index_curr - 1;
        if (index_prev >= i * matrix_size) {
            A[index_prev] = -htinvhsq;
        }
        // A_{i,j}
        A[index_curr] = 1 + 4.0*htinvhsq;
        // A_{i,j+1}
        index_next = index_curr + 1;
        if (index_next < (i + 1) * matrix_size) {
            A[index_next] = -htinvhsq;
        }
        // A_{i+1,j}
        index_down = index_curr + n;
        if (index_down < (i + 1) * matrix_size) {
            A[index_down] = -htinvhsq;
        }
    }
}


/**
 * @brief perform an elementwise division x ./ D
 * 
 * @param D diagonal matrix D stored as a vector
 * @param x result vector
 */
void elementwise_division_vector_2(double* D, double* x, int m, int n)
{
    int dim = m*n;
    for (int i = 0; i < dim; i++)
    {
        x[i] = x[i] / D[i];
    }
}

/**
 * @brief Given A = LDL', solve LDL'x = b
 * 
 */
void solveAxb_array(double *L, double *D, double *b, double *x, double *x_temp, int m, int n)
{
    forward_substitute_seq_array(b, L, x_temp, m, n);
    elementwise_division_vector_2(D, x_temp, m, n);
    backward_substitute_seq_array(x_temp, L, x, m, n);
}


/**
 * @brief Use Backward Euler method to solve heat equation
 * For each time step, we solve the linear system u_{k+1} = (I + ht*invhsq * A) \ (u_k + ht*invhsq * f);
 * 
 * @param f boudary correction term
 * @param u the heat map being updated
 * @param m 
 * @param n 
 */
void Backward_Euler_Array(double *A, double *L, double *D, double *f, double *uf, double *u, const int m, const int n)
{
    // u_{k+1} = (I + ht*invhsq*A) \ (u_k + ht*invhsq*f);
    double t = 0.0;
    const int MATRIX_DIM = m*n;

    // left side
    std::cout << "start init" << std::endl;
    initBackwardEulerMatrix_array(A, tau*invhsq, m, n); // initialize I + ht*invhsq*A

    std::cout << "start cholesky" << std::endl;
    ldlt_cholesky_decomposition_seq_array(A, L, D, m, n);
    // right side
    double u_temp[MATRIX_DIM];
    std::cout << "start running euler steps" << std::endl;
    while(t < endT)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                int index = i*n+j;
                uf[index] = u[index] + tau*invhsq*f[index];
            }
        }
        solveAxb_array(L, D, uf, u, u_temp, m, n);
        t += tau;
    }
}


