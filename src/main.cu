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
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <iostream>
#include "linalg_cu.h"
#include "global.h"
#include "debug_printing.h"

// void allocCSR(){
//     cudaMalloc(, sizeof(CSRMatrix));
// }


int main(int argc, char const *argv[])
{   
    const int m = 4;
    const int n = 4; 
    // const MATRIX_DIM = 1;
    double *u = (double *) malloc(m*n*sizeof(double));
    // loadCSV("../heat_map.csv", u, m*n);
    double *d = (double *) malloc(m*n*sizeof(double));
    double *temp_vec = (double *) malloc(m*n*sizeof(double));

    // Get a 2d Heat Map
    CSRMatrix A = CSRMatrix(m*n, 5*m*n);
    CSRMatrix L = CSRMatrix(m*n, 5*m*m*n); // store the L matrix
    CSRMatrix Lt = CSRMatrix(m*n, 5*m*m*n); // store the L matrix's transpose

    // malloc CSR Matrix A in GPU
    // A_values_d
    // A_columns_d
    // A_row_ptr_d
    double* A_values_d;
    int *A_columns_d, *A_row_ptr_d;
    cudaMalloc((void **)&A_values_d, 5*m*n*sizeof(double));
    cudaMalloc((void **)&A_columns_d, 5*m*n*sizeof(int));
    cudaMalloc((void **)&A_row_ptr_d, (m*n+1)*sizeof(int));
    // init A
    int initA_thread_x = 8;
    int initA_thread_y = 8;
    dim3 initA_grid(m/initA_thread_x+1, n/initA_thread_y+1, 1);
    dim3 initA_block(initA_thread_x, initA_thread_y, 1);
    initBackwardEulerMatrix_kernel<<<initA_grid, initA_block>>>(A_values_d, A_columns_d, A_row_ptr_d, tau*invhsq, m, n);
    cudaDeviceSynchronize();
    
    // // test init A
    // cudaMemcpy(A.values, A_values_d, 5*m*n*sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(A.columns, A_columns_d, 5*m*n*sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(A.row_ptr, A_row_ptr_d, (m*n+1)*sizeof(int), cudaMemcpyDeviceToHost);
    // A.rows = m*n;
    // print_csr_matrix(A);

    // malloc L and Lt in GPU
    // L_values_d, L_columns_d, L_row_ptr
    // Lt_values_d, Lt_columns_d, Lt_row_ptr
    double *L_values_d, *Lt_values_d;
    int *L_columns_d, *L_row_ptr_d, *Lt_columns_d, *Lt_row_ptr_d;
    cudaMalloc((void **)&L_values_d, 5*m*m*n*sizeof(double));
    cudaMalloc((void **)&L_columns_d, 5*m*m*n*sizeof(int));
    cudaMalloc((void **)&L_row_ptr_d, (m*n+1)*sizeof(int));
    cudaMalloc((void **)&Lt_values_d, 5*m*m*n*sizeof(double));
    cudaMalloc((void **)&Lt_columns_d, 5*m*m*n*sizeof(int));
    cudaMalloc((void **)&Lt_row_ptr_d, (m*n+1)*sizeof(int));
    // init L, LT\t
    int initL_thread = 64;
    dim3 initL_grid(m*n/initL_thread+1, 1, 1);
    dim3 initL_block(initL_thread, 1, 1);
    initL_kernel<<<initL_grid, initL_block>>>(L_values_d, L_columns_d, L_row_ptr_d, m, n);
    cudaDeviceSynchronize();
    initLt_kernel<<<initL_grid, initL_block>>>(Lt_values_d, Lt_columns_d, Lt_row_ptr_d, m, n);
    cudaDeviceSynchronize();

    // // test init L, Lt
    // cudaMemcpy(L.values, L_values_d, 5*m*m*n*sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(L.columns, L_columns_d, 5*m*m*n*sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(L.row_ptr, L_row_ptr_d, (m*n+1)*sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(Lt.values, Lt_values_d, 5*m*m*n*sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(Lt.columns, Lt_columns_d, 5*m*m*n*sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(Lt.row_ptr, Lt_row_ptr_d, (m*n+1)*sizeof(int), cudaMemcpyDeviceToHost);
    // L.rows = m*n;
    // Lt.rows = m*n;
    // print_csr_matrix(L);
    // print_csr_matrix(Lt);
    
    free(A_values_d);
    free(A_columns_d);
    free(A_row_ptr_d);
    free(L_values_d);
    free(L_columns_d);
    free(L_row_ptr_d);
    free(Lt_values_d);
    free(Lt_columns_d);
    free(Lt_row_ptr_d);

    
    
    // CSRMatrix d_A;
	// chol_kernel<<<grid,thread_block>>>(d_A.elements,time);


    return 0;
}
