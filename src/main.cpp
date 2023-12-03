/**
 * @file main.cpp
 * @author Xujin He (you@domain.com)
 * @brief main function for solving 2D heat equation, sequential code implementation
 * @version 0.1
 * @date 2023-11-29
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "linalg.h"

const double h = 0.01;
const double invhsq = 1/h/h;
const double tau = 0.01; // timestep size

int main(int argc, char const *argv[])
{   
    float* Data; // 2D heat map vector = (m*n)
    int DIM_X = 3; // grid dim = m
    int DIM_Y = 3; // grid dim = n
    int DATA_SIZE = DIM_X * DIM_Y * sizeof(float);

    int MATRIX_DIM = DIM_X * DIM_Y; // A = (mn x mn), this is extremely large so 

    // randomize a 2d Heat Map
    // srand(time(0));
    // for (i = 0; i < DIM_X*DIM_Y; i++) Data[i] = rand();

    // malloc CSR Matrix A in GPU
    // float* A;
    // cudaMalloc((void **)&A, MATRIX_DIM^2 * sizeof(float));
    CSRMatrix A = CSRMatrix(MATRIX_DIM, 5*DIM_X*DIM_Y);
    CSRMatrix L = CSRMatrix(MATRIX_DIM, 5*DIM_X*DIM_X*DIM_Y);
    double D[MATRIX_DIM];

    initializeCSRMatrix(A, DIM_X, DIM_Y);
    
    print_csr_matrix_info(A);
    print_csr_matrix(A);

    ldlt_cholesky_decomposition_seq(A, L, D);

    print_csr_matrix_info(L);
    print_csr_matrix(L);

    print_diagonal(D, MATRIX_DIM);

    // CSRMatrix Try = CSRMatrix(3, 6);
    // double value[] = {1,2,3,4,5,6};
    // int columns[] = {0,1,2,1,2,2};
    // int row_ptr[] = {0,3,5,6};
    // Try.values = value;
    // Try.row_ptr = row_ptr;
    // Try.columns = columns;
    // Try.rows = 3;
    // Try.non_zeros = 6;
    // print_csr_matrix_info(Try);
    // print_csr_matrix(Try);

    // solve for Ax = b; 
    double x[MATRIX_DIM];
    double x_temp[MATRIX_DIM];
    // backward_substitute(x_temp, L, x);
    // elementwise_division_vector(D, x, MATRIX_DIM);
    // forward_substitute(x, L, x_temp);

    
    return 0;
}
