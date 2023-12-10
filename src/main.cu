/**
 * @file main.cu
 * 
 * @author Xujin He (xh1131@nyu.edu)
 * @brief This is cuda code to run
 * @version 0.1
 * @date 2023-12-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <time.h>
#include <cuda.h>
#include <iostream>
#include <linalg_cu.h>

void allocCSR(){
    cudaMalloc(, sizeof(CSRMatrix));
}

int main(int argc, char const *argv[])
{   
    const int m = 76;
    const int n = 76; const MATRIX_DIM = 1;
    double *u = (double *) malloc(m*n*sizeof(double));
    loadCSV("../heat_map.csv", u, m*n);
    double *d = (double *) malloc(m*n*sizeof(double));
    double *temp_vec = (double *) malloc(m*n*sizeof(double));

    // Get a 2d Heat Map
    CSRMatrix A = CSRMatrix(m*n, 5*m*n);
    CSRMatrix L = CSRMatrix(m*n, 5*m*m*n); // store the L matrix
    CSRMatrix Lt = CSRMatrix(m*n, 5*m*m*n); // store the L matrix's transpose
    // malloc CSR Matrix A in GPU
    CSRMatrix d_A;
	chol_kernel<<<grid,thread_block>>>(d_A.elements,time);


    return 0;
}