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

__global__ void computeOffDiagonalL(double* A_values, int* A_columns, int* A_row_ptr, double* D, double* L_values, int* L_columns, int* L_row_ptr, int n);
__global__ void ldlt_colj_cu(int J, double *Lvalues, int *Lcolumns, int *Lrow_ptr, 
                             double *Avalues, int *Acolumns, int *Arow_ptr, double* D, const int col_len, const int row_len);
__global__ void ldlt_Dj_cu(double* D, int J, double *Lvalues, int *Lcolumns, int *Lrow_ptr, int n);
// __global__ void ldlt_cu();

void printCSR(double* L_values, int* L_columns, int* L_row_ptr, int M, int N){
    for (int i = 0; i < M*N; ++i) {
        int valueIndex = L_row_ptr[i];
        for (int j = 0; j < M*N; ++j) {
            if (valueIndex < L_row_ptr[i + 1] && L_columns[valueIndex] == j) {
                std::cout << L_values[valueIndex] << " ";
                ++valueIndex;
            } else {
                std::cout << " 0 ";
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
void printArray(double *D, int dim) {
    for (int i = 0; i < dim; i++)
    {
        std::cout << D[i] << std::endl;
    }
    
}
const int M = 4;
const int N = 4;
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
    // A_values_d
    // A_columns_d
    // A_row_ptr_d
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
    cudaMemcpy(L.values, L_values_d, 5*m*m*n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(L.columns, L_columns_d, 5*m*m*n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(L.row_ptr, L_row_ptr_d, (m*n+1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(Lt.values, Lt_values_d, 5*m*m*n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Lt.columns, Lt_columns_d, 5*m*m*n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(Lt.row_ptr, Lt_row_ptr_d, (m*n+1)*sizeof(int), cudaMemcpyDeviceToHost);
    L.rows = m*n;
    Lt.rows = m*n;

    const int grid_size = 1;
    const int block_size = 4;
    for (int J = 0; J < m*n; J++) // loop through all columns
    {
        // kernel update Djj
        ldlt_Dj_cu<<<grid_size,block_size>>>(D_d, J, L_values_d, L_columns_d, L_row_ptr_d, n);
        // kernel update Lij for all i > j
        if (J < n)
        {
            const int col_len = n;
            const int row_len = J;
            ldlt_colj_cu<<<grid_size,block_size>>>(J, L_values_d, L_columns_d, L_row_ptr_d, A_values_d, A_columns_d, A_row_ptr_d, D_d, col_len, row_len); 
        }
        else if (J > m*n-n)
        {
            const int col_len = m*n - J;
            const int row_len = n;
            ldlt_colj_cu<<<grid_size,block_size>>>(J, L_values_d, L_columns_d, L_row_ptr_d, A_values_d, A_columns_d, A_row_ptr_d, D_d, col_len, row_len);
        }
        else
        {
            const int col_len = n;
            const int row_len = n;
            ldlt_colj_cu<<<grid_size,block_size>>>(J, L_values_d, L_columns_d, L_row_ptr_d, A_values_d, A_columns_d, A_row_ptr_d, D_d, col_len, row_len);
        }
        // computeOffDiagonalL<<<grid_size,block_size>>>(J, A_values_d, A_columns_d, A_row_ptr_d, D_d, L_values_d, L_columns_d, L_row_ptr_d, n);
    }
    cudaMemcpy(d, D_d, m*n*sizeof(double), cudaMemcpyDeviceToHost);
    print_diagonal(d, m*n);
    cudaMemcpy(L.values, L_values_d, 5*m*m*n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(L.columns, L_columns_d, 5*m*m*n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(L.row_ptr, L_row_ptr_d, (m*n+1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(Lt.values, Lt_values_d, 5*m*m*n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Lt.columns, Lt_columns_d, 5*m*m*n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(Lt.row_ptr, Lt_row_ptr_d, (m*n+1)*sizeof(int), cudaMemcpyDeviceToHost);
    print_csr_matrix(A);
    print_csr_matrix_info(A);
    print_csr_matrix(L);
    print_csr_matrix_info(L);

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
    std::cout << "reached the end" << std::endl;
}

__device__ double get_ij(int* row_ptr, int* columns, double *values, int i, int j) 
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
__device__ void set_ij(int* row_ptr, int* columns, double *values, int i, int j, double value) 
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
__global__ void ldlt_colj_cu(int J, double *Lvalues, int *Lcolumns, int *Lrow_ptr, 
                             double *Avalues, int *Acolumns, int *Arow_ptr, double* D, const int col_len, const int row_len)
{    
    // Assume L and Lt have zeros pre-allocated
    // const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // const unsigned int total_t = blockDim.x * gridDim.x;
    
    // assuming total_t << N
    // we let kth thread handle from row k*bh to (k+1)*bh
    // the last thread will handle the middle rows
    // const int col_len = N;            // total number of entries to update ;for starting & interior cols, col length = N; but for J>M*N-N it is (M*N-J) 
    const double DJ = D[J];

    // load shared memory
    // extern __shared__ double Lj_[];
    // const int thread_load = row_len/blockDim.x; 

    // for (int i = 0; i < thread_load; i++)
    // {
    //     Lj_[threadIdx.x*thread_load + i] = Lvalues[Lrow_ptr[J] + threadIdx.x*thread_load + i];
    // }
    // if (threadIdx.x == 0) {
    //     for (int i = thread_load*blockDim.x; i < row_len; i++) Lj_[i] = Lvalues[Lrow_ptr[J] + i];
    // }
    
    __syncthreads();

    for (int local_i = 0; local_i < col_len; local_i++)
    {
        const int global_i = J + 1 + blockIdx.x + gridDim.x * local_i;
        
        if (global_i < J + col_len + 1) 
        {
            const int ith_row_len = row_len - gridDim.x * local_i - blockIdx.x;
            for (int local_k = 0; local_k < ith_row_len; local_k++)
            {
                const int global_k = blockDim.x * local_k + threadIdx.x;
                if (global_k < J) {
                    const int Lik = get_ij(Lrow_ptr, Lcolumns, Lvalues, global_i, global_k);
                    const int Ljk = get_ij(Lrow_ptr, Lcolumns, Lvalues, J, global_k);
                    const int Dk = D[global_k];
                    set_ij(Lrow_ptr, Lcolumns, Lvalues, global_i, J, Lik * Ljk * Dk); // sum_{k=0}^{j-1} Lik * Ljk 
                }
            }
            __syncthreads();
            if (threadIdx.x == 0)
            {
                const double Aij = get_ij(Arow_ptr, Acolumns, Avalues, global_i, J);
                const double value = (Aij - get_ij(Lrow_ptr, Lcolumns, Lvalues, global_i, J)) / DJ;
                set_ij(Lrow_ptr, Lcolumns, Lvalues, global_i, J, value); // sum_{k=0}^{j-1} Lik * Ljk 
            }  
        }      
    }

    // for (local_i=0; local_i<th; local_i++) // loop the rows, assuming total_t*th >= col_len so that all entries Lijs are covered
    // {
    //     double sumL2 = 0;
    //     const int global_i = J + th*tid + local_i + 1; // computing row global_i, whose value ranges from Lrow_ptr[global_i] to Lrow_ptr[global_i]+N+j-global_i, 
    //     if (global_i < N + 1) 
    //     {
    //         int x = Lrow_ptr[global_i];
    //         // for (int k = 0; k < J; k++)
    //         for (int k = 0; k < J; k++) 
    //         {
    //             // double Lik = Lvalues[x+k]; 
    //             // double Ljk = Lvalues[Lrow_ptr[J]+k];
    //             double Lik = get_ij(Lrow_ptr, Lcolumns, Lvalues, global_i, k);
    //             double Ljk = get_ij(Lrow_ptr, Lcolumns, Lvalues, J, k);
    //             sumL2 += Lik * Ljk * D[k];
    //         }
    //         // set Lij = (Aij - sumL2) / Dj, need to figure out if Aij = 0 or not
    //         double Aij = 0;
    //         for (int p = Arow_ptr[global_i]; p < Arow_ptr[global_i+1]; p++)
    //         {
    //             if (J == Acolumns[p]) Aij = Avalues[p];
    //         }
    //         Lvalues[x+J] = (Aij - sumL2) / DJ;
    //     } 
    //     else if (global_i < J + col_len + 1) // middle cols 
    //     {
    //         int x = Lrow_ptr[global_i];
    //         // for (int k = 0; k < J-Lcolumns[x]; k++) 
    //         for (int k = J - N; k < J; k++) 
    //         {
    //             // double Lik = Lvalues[x+k]; 
    //             // double Ljk = Lvalues[Lrow_ptr[J]+k+global_i-J];
    //             double Lik = get_ij(Lrow_ptr, Lcolumns, Lvalues, global_i, k);
    //             double Ljk = get_ij(Lrow_ptr, Lcolumns, Lvalues, J, k);
    //             sumL2 += Lik * Ljk * D[k+global_i-J];
    //         }
    //         // set Lij = (Aij - sumL2) / DJ, need to figure out if Aij = 0 or not
    //         double Aij = 0;
    //         for (int p = Arow_ptr[global_i]; p < Arow_ptr[global_i+1]; p++)
    //         {
    //             if (J == Acolumns[p]) Aij = Avalues[p];
    //         }
    //         Lvalues[x+J-global_i+N] = (Aij - sumL2) / DJ;
    //     }
    // }
}

__global__ void ldlt_Dj_cu(double* D, int J, double *Lvalues, int *Lcolumns, int *Lrow_ptr, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < Lrow_ptr[J+1] - Lrow_ptr[J])
    {
        int k = Lrow_ptr[J] + tid;
        double Ljk = Lvalues[k];
        atomicAdd(&D[J], -Ljk * Ljk * D[Lcolumns[k]]); // sum_k Ljk^2 * Dk
    }
    
    __syncthreads();

    if (tid == 0) {
        const double AJJ = 4;
        atomicAdd(&D[J], AJJ);
        Lvalues[Lrow_ptr[J+1]-1] = 1; // Ljj = 1;
    }
}
