#include <time.h>
// #include <cuda.h>
#include <iostream>

#include "csr.h"

const double h = 0.01;
const double invhsq = 1/h/h;
const double tau = 0.01; // timestep size

/**
 * @brief forward_substitute to solve x = b \ L, L lower triangular in CSR format
 * 
 */
// void forward_substitute(float* V, unsigned* rowInd, unsigned colInd, float* b, float* x, unsigned len);
// {
//     x[0] = b[0] / V[0];
//     for (unsigned i = 0; i < len-1; i++)
//     {
//         float sum;
//         int vindex = rowInd[i];
//         for (unsigned j = vindex; j < rowInd[i+1]; j++)
//         {
//             int bcol = colInd[j];
//             sum += V[vindex] * b[bcol];
//         }
//         x[i] = b[i] - sum / V[i][i];
//     }
// }

int main(int argc, char const *argv[])
{   
    float* Data; // 2D heat map vector = (m*n)
    int DIM_X = 3; // grid dim = m
    int DIM_Y = 3; // grid dim = n
    int DATA_SIZE = DIM_X * DIM_Y * sizeof(float);

    int MATRIX_DIM = DIM_X * DIM_X * DIM_Y * DIM_Y; // A = (mn x mn), this is extremely large so 

    // randomize a 2d Heat Map
    // srand(time(0));
    // for (i = 0; i < DIM_X*DIM_Y; i++) Data[i] = rand();

    // malloc CSR Matrix A in GPU
    // float* A;
    // cudaMalloc((void **)&A, MATRIX_DIM * sizeof(float));
    CSRMatrix A = CSRMatrix(DIM_X*DIM_Y, 5*DIM_X*DIM_Y);
    CSRMatrix L = CSRMatrix(DIM_X*DIM_Y, 5*DIM_X*DIM_X*DIM_Y);
    double* D;

    initializeCSRMatrix(A, DIM_X, DIM_Y);
    ldlt_cholesky_decomposition_seq(A, L, D);

    // print_csr_matrix(A);
    // print_csr_matrix_info(A);

    print_csr_matrix(L);
    
    return 0;
}
