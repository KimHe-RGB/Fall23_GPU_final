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
#include "debug_printing.h"
#include "linalg_cu.h"

#ifndef __GLOBAL_H
#include "global.h"
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void ldlt_colj_cu(int J, double *Lvalues, int *Lcolumns, int *Lrow_ptr, 
                             double *Avalues, int *Acolumns, int *Arow_ptr, double* D, const int col_len, const int row_len);
__global__ void ldlt_Dj_cu(double* D, int J, double *Lvalues, int *Lcolumns, int *Lrow_ptr);

const int M = 5;
const int N = 5;

int main(int argc, char const *argv[]){

    const int m = M;
    const int n = N; 
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
    double* A_values_d;
    int *A_columns_d, *A_row_ptr_d;
    double *D_d;
    cudaMalloc((void **)&A_values_d, 5*m*n*sizeof(double));
    cudaMalloc((void **)&A_columns_d, 5*m*n*sizeof(int));
    cudaMalloc((void **)&A_row_ptr_d, (m*n+1)*sizeof(int));
    cudaMalloc((void **)&D_d, m*n*sizeof(double));
    // init A
    int initA_thread_x = 8;
    int initA_thread_y = 8;
    dim3 initA_grid(m/initA_thread_x+1, n/initA_thread_y+1, 1);
    dim3 initA_block(initA_thread_x, initA_thread_y, 1);
    initBackwardEulerMatrix_kernel<<<initA_grid, initA_block>>>(A_values_d, A_columns_d, A_row_ptr_d, tau*invhsq, m, n);
    cudaDeviceSynchronize();
    
    // test init A
    cudaMemcpy(A.values, A_values_d, 5*m*n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(A.columns, A_columns_d, 5*m*n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(A.row_ptr, A_row_ptr_d, (m*n+1)*sizeof(int), cudaMemcpyDeviceToHost);
    A.rows = m*n;

    // malloc L and Lt in GPU
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
    cudaMemcpy(L.values, L_values_d, 5*m*m*n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(L.columns, L_columns_d, 5*m*m*n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(L.row_ptr, L_row_ptr_d, (m*n+1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(Lt.values, Lt_values_d, 5*m*m*n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Lt.columns, Lt_columns_d, 5*m*m*n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(Lt.row_ptr, Lt_row_ptr_d, (m*n+1)*sizeof(int), cudaMemcpyDeviceToHost);
    L.rows = m*n;
    Lt.rows = m*n;

    const int block_size = 5;
    for (int J = 0; J < m*n; J++) // loop through all columns
    {
        const int row_len = L.row_ptr[J+1] - L.row_ptr[J]; 
        // kernel update Djj
        // compute sumL
        // double Ajj = M_get_ij(A.row_ptr, A.columns, A.values, J, J);
        // Dj_kernel<<<1, J+1>>>(D_d, J,  L_values_d, L_columns_d, L_row_ptr_d, Ajj);
        cudaDeviceSynchronize();
        ldlt_Dj_cu<<<1,row_len>>>(D_d, J, L_values_d, L_columns_d, L_row_ptr_d);
        // kernel update Lij for all i > j
        if (J < n)
        {
            const int col_len = n;
            const int row_len = J;
            ldlt_colj_cu<<<col_len,block_size>>>(J, L_values_d, L_columns_d, L_row_ptr_d, A_values_d, A_columns_d, A_row_ptr_d, D_d, col_len, row_len); 
        }
        else if (J > m*n-n)
        {
            const int col_len = m*n - J;
            const int row_len = n;
            ldlt_colj_cu<<<col_len,block_size>>>(J, L_values_d, L_columns_d, L_row_ptr_d, A_values_d, A_columns_d, A_row_ptr_d, D_d, col_len, row_len);
        }
        else
        {
            const int col_len = n;
            const int row_len = n;
            ldlt_colj_cu<<<col_len,block_size>>>(J, L_values_d, L_columns_d, L_row_ptr_d, A_values_d, A_columns_d, A_row_ptr_d, D_d, col_len, row_len);
        }
    }
    cudaMemcpy(d, D_d, m*n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(L.values, L_values_d, 5*m*m*n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(L.columns, L_columns_d, 5*m*m*n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(L.row_ptr, L_row_ptr_d, (m*n+1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(Lt.values, Lt_values_d, 5*m*m*n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Lt.columns, Lt_columns_d, 5*m*m*n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(Lt.row_ptr, Lt_row_ptr_d, (m*n+1)*sizeof(int), cudaMemcpyDeviceToHost);
    print_diagonal(d, m*n);
    // print_csr_matrix(A);
    // print_csr_matrix_info(A);
    print_csr_matrix(L);
    // print_csr_matrix_info(L);

    // Boundary Condition terms
    double* f_d;
    cudaMalloc((void **)&f_d, m*n*sizeof(double));
    int BCthread_x = 8;
    int BCthread_y = 8;
    dim3 BCgrid(m/BCthread_x+1, n/BCthread_y+1, 1);
    dim3 BCblock(BCthread_x, BCthread_y, 1);
    BoundaryCondition_kernel<<<BCgrid, BCblock>>>(f_d, m, n, h);
    cudaDeviceSynchronize();

    // Backward Euler steps
    int BEthread = 64;
    int BEblock = m*n/BEthread + 1;
    int total_steps = 1; //endT/tau;
    // allocate memory to store u and b on device
    double *u_d, *b_d;
    cudaMalloc((void **)&u_d, m*n*sizeof(double));
    cudaMalloc((void **)&b_d, m*n*sizeof(double));
    cudaMemcpy(u_d, u, m*n*sizeof(double), cudaMemcpyHostToDevice);
    for (int p = 0; p < total_steps; p++)
    {
        // launch kernel to compute updated b
        Updateb_kernel<<<BEblock, BEthread>>>(b_d, u_d, f_d, tau*invhsq, m*n);
        cudaDeviceSynchronize();
        // solveAxb(L, Lt, D, b, u, MATRIX_DIM); TBA
    }
    
    cudaFree(A_values_d);
    cudaFree(A_columns_d);
    cudaFree(A_row_ptr_d);
    cudaFree(L_values_d);
    cudaFree(L_columns_d);
    cudaFree(L_row_ptr_d);
    cudaFree(Lt_values_d);
    cudaFree(Lt_columns_d);
    cudaFree(Lt_row_ptr_d);
    cudaFree(u_d);
    cudaFree(f_d);
    cudaFree(b_d);
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
        exit(-1);
    }
}

__device__ double get_ij(int* row_ptr, int* columns, double *values, const int i, const int j) 
{
    int start = row_ptr[i];
    int end = row_ptr[i + 1];
    for (int k = start; k < end; k++) {
        if (columns[k] == j) {
            return values[k];
        }
    }
    // Column not found, as columns are in ascending order
    return 0;
}
__device__ void set_ij(int* row_ptr, int* columns, double *values, const int i, const int j, double value) 
{
    int start = row_ptr[i];
    int end = row_ptr[i + 1];
    // Check if the element already exists in the matrix
    for (int k = start; k < end; k++) {
        if (columns[k] == j) {
            values[k] = value;  // Update existing value
            return;
        }
    }
    return;
}

#define get_ij_direct(row_ptr, columns, values, i, j) values[row_ptr[i] + j];

/**
 * @brief 
 * 
 * @param J global column index we are computing
 * @param Lvalues 
 * @param Lcolumns 
 * @param Lrow_ptr 
 * @param Avalues 
 * @param Acolumns 
 * @param Arow_ptr 
 * @param D 
 * @return __global__ 
 */
__global__ void ldlt_colj_cu(const int J, double *Lvalues, int *Lcolumns, int *Lrow_ptr, 
                             double *Avalues, int *Acolumns, int *Arow_ptr, double* D, const int col_len, const int row_len)
{    
    // Assume L and Lt have zeros pre-allocated
    // const int col_len = N;            // total number of entries to update ;for starting & interior cols, col length = N; but for J>M*N-N it is (M*N-J) 
    const double DJ = D[J];

    // load shared memory
    extern __shared__ double suml2[];
    suml2[threadIdx.x] = 0;
    __syncthreads();
    
    // gridDim.x = col_len; j + col_length is bottom non-0 element's global row index
    const int global_i = J + 1 + blockIdx.x; 
    int ith_row_len = row_len - 1 - blockIdx.x;
    if (J < N) ith_row_len = row_len;

    for (int local_k = threadIdx.x; local_k < ith_row_len; local_k+=blockDim.x)
    {
        const int global_k = global_i - col_len + local_k;
        // const int kk = (1 + blockIdx.x) + local_k;
        if (global_k >= 0 && global_k < J) {
            const double Lik = get_ij(Lrow_ptr, Lcolumns, Lvalues, global_i, global_k);
            const double Ljk = get_ij(Lrow_ptr, Lcolumns, Lvalues, J, global_k);
            // const double Ljk = get_ij_direct(Lrow_ptr, Lcolumns, Lvalues, J, global_k);
            const double Dk = D[global_k];
            suml2[threadIdx.x] += Lik * Ljk * Dk;
        }
    }
    __syncthreads();
    // reduction by sequential addressing
    for (int s = blockDim.x/2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            suml2[threadIdx.x] += suml2[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        const double Aij = get_ij(Arow_ptr, Acolumns, Avalues, global_i, J);
        const double value2 = (Aij - suml2[0]) / DJ;
        set_ij(Lrow_ptr, Lcolumns, Lvalues, global_i, J, value2); // sum_{k=0}^{j-1} Lik * Ljk
        // if (J == M*N-3) {
        //     double x = get_ij(Lrow_ptr, Lcolumns, Lvalues, global_i, 12);
        //     double x2 = get_ij(Lrow_ptr, Lcolumns, Lvalues, J, 12); const double xx = x*x2*D[12];
        //     double y = get_ij(Lrow_ptr, Lcolumns, Lvalues, global_i, 11);
        //     double y2 = get_ij(Lrow_ptr, Lcolumns, Lvalues, J, 11);const double yy = y*y2*D[11];
        //     double z = get_ij(Lrow_ptr, Lcolumns, Lvalues, global_i, 10);
        //     double z2 = get_ij(Lrow_ptr, Lcolumns, Lvalues, J, 10);const double zz = z*z2*D[10];
        //     double v = (Aij - xx - yy - zz) / DJ;
        //     set_ij(Lrow_ptr, Lcolumns, Lvalues, global_i, J, 114); // sum_{k=0}^{j-1} Lik * Ljk
        // }
    } 
}

__global__ void ldlt_Dj_cu(double* D, int J, double *Lvalues, int *Lcolumns, int *Lrow_ptr)
{
    // load into shared
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int global_k = Lcolumns[Lrow_ptr[J]] + threadIdx.x;

    const double Ljk = Lvalues[Lrow_ptr[J]+tid];
    sdata[tid] = Ljk * Ljk * D[global_k];

    __syncthreads();
    // reduction by sequential addressing
    for (int s = blockDim.x/2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write back
    if (tid == 0) {
        const double AJJ = 4;
        D[J] = AJJ - sdata[0];
        Lvalues[Lrow_ptr[J+1]-1] = 1; // Ljj = 1;
    }
}
