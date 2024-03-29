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
#include <chrono>
#include <stdio.h>

int main(int argc, char const *argv[])
{   
    if (argc != 3)
    {
        printf("heat_seq m n\n");
    }

    unsigned int m = (unsigned int) atoi(argv[1]); // grid dim = m
    unsigned int n = (unsigned int) atoi(argv[1]); // grid dim = n
    const int MATRIX_DIM = m * n; // A = (mn x mn), this is extremely large

    // Init u
    double *u = (double *) malloc(m*n*sizeof(double));
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            u[i*n+j] = 0;
        }
    }

    // // malloc CSR Matrix A in CPU
    // CSRMatrix A = CSRMatrix(MATRIX_DIM, 5*m*n);
    // CSRMatrix L = CSRMatrix(MATRIX_DIM, 5*m*m*n); // store the L matrix
    // CSRMatrix Lt = CSRMatrix(MATRIX_DIM, 5*m*m*n); // store the L matrix's transpose
    // double *D = (double *) malloc(m*n*sizeof(double)); // store the diagonal matrix

    // initializeCSRMatrix(A, m, n);
    // initBackwardEulerCSRMatrix(A, tau*invhsq, m, n);

    // // Test: init
    // print_csr_matrix_info(A);
    // print_csr_matrix(A);

    // ldlt_cholesky_decomposition_seq(A, L, Lt, D, m, n);
    // // Test: Cholesky works
    // print_csr_matrix_info(L);
    // print_csr_matrix(L);
    // print_csr_matrix_info(Lt);
    // print_csr_matrix(Lt);
    // print_diagonal(D, MATRIX_DIM); // [4.0000, 3.7500, 3.7333, 3.4286, 3.3333, 3.3000, 3.2386, 3.1978, 3.1723]

    // // Test: Solving Ax = b by Cholesky, m=n=3
    // double b[] = {0.8147, 0.9058, 0.1270, 0.9134, 0.6324, 0.0975, 0.2785, 0.5469, 0.9575};
    // double x[MATRIX_DIM]; double temp_vec[MATRIX_DIM];
    // solveAxb(L, Lt, D, b, x, MATRIX_DIM);
    // print_diagonal(x, MATRIX_DIM); // [0.6397, 0.7878, 0.6533, 0.9564, 0.9524, 0.7419, 0.6670, 0.6910, 0.5976]

    // Test: load heat map from csv:
    // example initial condition is 76 x 76
    // const int dim = 8;
    // double u[] = {0.8147, 0.9058, 0.1270, 0.9134, 0.6324, 0.0975, 0.2785, 0.5469, 
    //                 0.9575, 0.9649, 0.1576, 0.9706, 0.9572, 0.4854, 0.8003, 0.1419, 
    //                 0.4218, 0.9157, 0.7922, 0.9595, 0.6557, 0.0357, 0.8491, 0.9340, 
    //                 0.6787, 0.7577, 0.7431, 0.3922, 0.6555, 0.1712, 0.7060, 0.0318, 
    //                 0.2769, 0.0462, 0.0971, 0.8235, 0.6948, 0.3171, 0.9502, 0.0344, 
    //                 0.4387, 0.3816, 0.7655, 0.7952, 0.1869, 0.4898, 0.4456, 0.6463, 
    //                 0.7094, 0.7547, 0.2760, 0.6797, 0.6551, 0.1626, 0.1190, 0.4984, 
    //                 0.9597, 0.3404, 0.5853, 0.2238, 0.7513, 0.2551, 0.5060, 0.6991};
    // double u[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    // const int dim = 76;
    // double *u = (double *) malloc(dim*dim*sizeof(double));
    // loadCSV("../heat_map.csv", u, dim*dim);

    // Backward Euler steps
    // print_heat_map(u, m, n);
    auto start = std::chrono::high_resolution_clock::now();
    Backward_Euler_CSR(u, m, n);
    auto end = std::chrono::high_resolution_clock::now();
    // print_heat_map(u, m, n);
    std::chrono::duration<double> duration = end - start;
    double time_taken = duration.count();
    printf("%lf\n", time_taken);
    
    
    snapshot("../outcome.gif", u, m, n, 1);

    // writeCSV("../heat_map_out.csv", u, m, n);
    return 0;
}
