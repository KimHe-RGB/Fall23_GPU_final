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

/**
 * @brief 
 * 
 */
void Backward_Euler()
{

}

/**
 * @brief Given A = LDL', solve LDL'x = b
 * 
 */
void solveAxb(CSRMatrix &L, CSRMatrix &Lt, double *D, double *b, double *x, int MATRIX_DIM)
{
    double x_temp[MATRIX_DIM];
    forward_substitute(b, L, x_temp);
    // print_diagonal(x_temp, MATRIX_DIM);
    elementwise_division_vector(D, x_temp, MATRIX_DIM);
    // print_diagonal(x_temp, MATRIX_DIM);
    backward_substitute(x_temp, Lt, x);
    // print_diagonal(x, MATRIX_DIM);
}

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
    CSRMatrix Lt = CSRMatrix(MATRIX_DIM, 5*DIM_X*DIM_X*DIM_Y);
    double D[MATRIX_DIM];

    initializeCSRMatrix(A, DIM_X, DIM_Y);
    // Test: init
    // print_csr_matrix_info(A);
    // print_csr_matrix(A);

    ldlt_cholesky_decomposition_seq(A, L, Lt, D);
    // Test: Cholesky works
    // print_csr_matrix_info(L);
    // print_csr_matrix(L);
    // print_csr_matrix_info(Lt);
    // print_csr_matrix(Lt);
    // print_diagonal(D, MATRIX_DIM);

    // Test: Solving Ax = b by Cholesky
    double b[] = {0.8147, 0.9058, 0.1270, 0.9134, 0.6324, 0.0975, 0.2785, 0.5469, 0.9575};
    double x[MATRIX_DIM]; 

    solveAxb(L, Lt, D, b, x, MATRIX_DIM);
    
    return 0;
}
