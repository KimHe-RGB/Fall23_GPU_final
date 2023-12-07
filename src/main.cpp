/**
 * @file main.cpp
 * @author Xujin He (xh1131@nyu.edu)
 * @brief main function for solving 2D heat equation, sequential code implementation
 * @version 0.1
 * @date 2023-11-29
 * 
 * @copyright Copyright (c) 2023
 */

#include "global.h"
#include "linalg.h"
#include "load_heat_map.h"

int main(int argc, char const *argv[])
{   
    float* Data; // 2D heat map vector = (m*n)
    const int DIM_X = 3; // grid dim = m
    const int DIM_Y = 3; // grid dim = n
    int DATA_SIZE = DIM_X * DIM_Y * sizeof(float);

    const int MATRIX_DIM = DIM_X * DIM_Y; // A = (mn x mn), this is extremely large

    // malloc CSR Matrix A in CPU
    // CSRMatrix A = CSRMatrix(MATRIX_DIM, 5*DIM_X*DIM_Y);
    // CSRMatrix L = CSRMatrix(MATRIX_DIM, 5*DIM_X*DIM_X*DIM_Y);
    // CSRMatrix Lt = CSRMatrix(MATRIX_DIM, 5*DIM_X*DIM_X*DIM_Y);
    // double D[MATRIX_DIM];
    // double temp_vec[MATRIX_DIM];

    // initializeCSRMatrix(A, DIM_X, DIM_Y);
    // // initBackwardEulerMatrix(A, tau*invhsq, DIM_X, DIM_Y);
    // // Test: init
    // print_csr_matrix_info(A);
    // print_csr_matrix(A);

    // ldlt_cholesky_decomposition_seq(A, L, Lt, D, DIM_X, DIM_Y);
    // // Test: Cholesky works
    // print_csr_matrix_info(L);
    // print_csr_matrix(L);
    // print_csr_matrix_info(Lt);
    // print_csr_matrix(Lt);
    // print_diagonal(D, MATRIX_DIM); // [4.0000, 3.7500, 3.7333, 3.4286, 3.3333, 3.3000, 3.2386, 3.1978, 3.1723]

    // // Test: Solving Ax = b by Cholesky
    // double b[] = {0.8147, 0.9058, 0.1270, 0.9134, 0.6324, 0.0975, 0.2785, 0.5469, 0.9575};
    // double x[MATRIX_DIM]; 
    // solveAxb(L, Lt, D, b, x, temp_vec, MATRIX_DIM);
    // print_diagonal(x, MATRIX_DIM); // [0.6397, 0.7878, 0.6533, 0.9564, 0.9524, 0.7419, 0.6670, 0.6910, 0.5976]

    // Test: load heat map from csv:
    // example initial condition is 76 x 76
    const int dim = 76;
    double u[dim*dim];
    loadCSV("../heat_map.csv", u, dim*dim);
    double d[dim*dim];
    double temp_vec[dim*dim];
    
    // initialize boundary condition correction term
    double f[dim*dim];
    // Backward Euler steps
    Backward_Euler_CSR(f, u, d, temp_vec, dim, dim);
    print_diagonal(u, dim*dim);


    // double A[dim*dim*dim*dim];
    // double L[dim*dim*dim*dim];
    // double D[dim*dim];
    // double uf[dim*dim];
    // Backward_Euler_Array(A, L, D, f, uf, u, dim, dim);
    // print_diagonal(u, dim*dim);

    // writeCSV("../heat_map_out.csv", u, dim, dim);
    return 0;
}
