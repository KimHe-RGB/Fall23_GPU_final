#include <cuda.h>

/**
 * @brief cuda kernel to init Backward Euler Matrix A
 */
__global__ void initBackwardEulerMatrix_kernel(double* A_values, int* A_columns, int* A_row_ptr, double htinvhsq, int m, int n)
{
    // Initialize CSR matrix values and structure
    
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    int row = i*n+j;
    int count = 0;
    // double center = 1+4.0*htinvhsq; 
    // double other = -htinvhsq;
    double center = 4.0; 
    double other = -1;
    if (i == 0) // first block
    {
        count = 0;
        if (j == 0) // first row
        {
            A_row_ptr[row] = count;
            A_values[count] = center;
            A_columns[count] = row;
            A_values[count+1] = other;
            A_columns[count+1] = row+1;
            A_values[count+2] = other;
            A_columns[count+2] = row+n;
        }
        count = 3 + 4*(j-1);
        if (j > 0 && j < n-1)
        {
            A_row_ptr[row] = count;
            A_values[count] = other;
            A_columns[count] = row-1;
            A_values[count+1] = center;
            A_columns[count+1] = row;
            A_values[count+2] = other;
            A_columns[count+2] = row+1;
            A_values[count+3] = other;
            A_columns[count+3] = row+n;
        }
        else if (j == n-1)
        {
            A_row_ptr[row] = count;
            A_values[count] = other;
            A_columns[count] = row-1;
            A_values[count+1] = center;
            A_columns[count+1] = row;
            A_values[count+2] = other;
            A_columns[count+2] = row+n;
        }
    }
    else if(i < m-1)
    {
        count = (i-1) * (5*n-2) + 4*n-2;
        if (j == 0) 
        {
            A_row_ptr[row] = count;
            A_values[count] = other;
            A_columns[count] = row-n;
            A_values[count+1] = center;
            A_columns[count+1] = row;
            A_values[count+2] = other;
            A_columns[count+2] = row+1;
            A_values[count+3] = other;
            A_columns[count+3] = row+n;
        }
        count += 4 + 5*(j-1);
        if (j > 0 && j < n-1)
        {
            A_row_ptr[row] = count;
            A_values[count] = other;
            A_columns[count] = row-n;
            A_values[count+1] = other;
            A_columns[count+1] = row-1;
            A_values[count+2] = center;
            A_columns[count+2] = row;
            A_values[count+3] = other;
            A_columns[count+3] = row+1;
            A_values[count+4] = other;
            A_columns[count+4] = row+n;
        }
        else if (j == n-1)
        {
            A_row_ptr[row] = count;
            A_values[count] = other;
            A_columns[count] = row-n;
            A_values[count+1] = other;
            A_columns[count+1] = row-1;
            A_values[count+2] = center;
            A_columns[count+2] = row;
            A_values[count+3] = other;
            A_columns[count+3] = row+n;
        }
    }
    else if (i == m-1) // last block
    {
        count = (m-2) * (5*n-2) + 4*n-2; 
        if (j == 0) 
        {
            A_row_ptr[row] = count;
            A_values[count] = other;
            A_columns[count] = row-n;
            A_values[count+1] = center;
            A_columns[count+1] = row;
            A_values[count+2] = other;
            A_columns[count+2] = row+1;
        }
        count += 3 + 4*(j-1);
        if (j > 0 && j < n-1)
        {
            A_row_ptr[row] = count;
            A_values[count] = other;
            A_columns[count] = row-n;
            A_values[count+1] = other;
            A_columns[count+1] = row-1;
            A_values[count+2] = center;
            A_columns[count+2] = row;
            A_values[count+3] = other;
            A_columns[count+3] = row+1;
        }
        else if (j == n-1) // last row
        {
            A_row_ptr[row] = count;
            A_values[count] = other;
            A_columns[count] = row-n;
            A_values[count+1] = other;
            A_columns[count+1] = row-1;
            A_values[count+2] = center;
            A_columns[count+2] = row;
            A_row_ptr[row+1] = count+3;
        }
    }
}

/**
 * @brief cuda kernel to compute forward substitution
 */
__global__ void forward_substitute_cu(const double* b, const CSRMatrix& L, double* x)
{ 
    int rows = L.rows;

    // Forward substitution
    for (int i = 0; i < rows; ++i) {
        // Compute the sum of L(i, j) * x(j) for j = 0 to i-1
        double sum = 0.0;
        for (int k = L.row_ptr[i]; k < L.row_ptr[i + 1]; ++k) {
            int j = L.columns[k];
            sum += L.values[k] * x[j];
        }

        // Solve for x(i)
        x[i] = (b[i] - sum) / L.values[L.row_ptr[i + 1] - 1];
    }
}

/**
 * @brief cuda kernel to compute backward substitution
 * backward substitute to solve x = b \ Lt
 */
__global__ void backward_substitute_cu(const double* b, const CSRMatrix& U, double* x)
{
    int rows = U.rows;
    // Backward substitution
    for (int i = rows - 1; i >= 0; i--) {
        // Compute the sum of U(i, j) * x(j) for j = i+1 to rows-1
        double sum = 0.0;
        for (int k = U.row_ptr[i] + 1; k < U.row_ptr[i + 1]; ++k) {
            int j = U.columns[k];
            sum += U.values[k] * x[j];
        }
        // Solve for x(i)
        x[i] = (b[i] - sum) / U.values[U.row_ptr[i]];
    }
}

/**
 * @brief cuda kernel to perform an elementwise division x ./ D
 * 
 * @param D diagonal matrix D stored as a vector
 * @param x result vector
 */
__global__ void elementwise_division_cu(double* D, double* x, int dim)
{
    
    for (int i = 0; i < dim; i++)
    {
        x[i] = x[i] / D[i];
    }
}

/**
 * @brief Pre-allocate non-zero position of L and Lt in CSR Format
 * 
 * @param L 
 * @param Lt 
 * @param m 
 * @param n 
 * @return __global__ 
 */
__global__ void initL_kernel(double* L_values, int* L_columns, int* L_row_ptr, int m, int n)
{
    // Initialize L matrix 
    // to avoid too many shifting, we pre-allocate the non-zero terms with zero

    int row = (blockIdx.x * blockDim.x) + threadIdx.x;
    int count;

    // row 0 has 1 elements
    if (row == 0)
    {
        L_row_ptr[row] = 0;
        L_values[row] = 0;
        L_columns[row] = 0;
    }
    // row 1 to n-1: row i has (i+1) elements
    else if (row < n)
    {
        count = row*(row+1)/2;
        L_row_ptr[row] = count;
        for (int i = 0; i < row+1; i++)
        {
            L_values[count] = 0;
            L_columns[count] = i;
            count++;
        }
    }
    // row n to m*n-1: row i has (n+1) elements
    else if(row < m*n)
    {
        count = n*(n+1)/2 + (row-n)*(n+1);
        L_row_ptr[row] = count;
        for (int i = 0; i < n+1; i++)
        {
            L_values[count] = 0;
            L_columns[count] = row-n+i;
            count++;
        }
        if (row == m*n-1) // last line
        {
            L_row_ptr[row+1] = count;
        }
    }
}
__global__ void initLt_kernel(double* Lt_values, int* Lt_columns, int* Lt_row_ptr, int m, int n)
{
    // Initialize Lt matrix 
    // to avoid too many shifting, we pre-allocate the non-zero terms with zero

    int row = (blockIdx.x * blockDim.x) + threadIdx.x;
    int count;

    // row 0 to (m-1)*n-1: row i has (n+1) elements
    if (row < (m-1)*n)
    {
        count = row*(n+1);
        Lt_row_ptr[row] = count;
        for (int i = 0; i < n+1; i++)
        {
            Lt_values[count] = 0;
            Lt_columns[count] = row+i;
            count++;
        }
    }
    // row (m-1)*n to m*n-1: row i has (m*n-i) elements
    else if (row < m*n)
    {
        count = (m-1)*n*(n+1) + (n+m*n-row+1)*(n-m*n+row)/2;
        Lt_row_ptr[row] = count;
        for (int i = 0; i < m*n-row; i++)
        {
            Lt_values[count] = 0;
            Lt_columns[count] = row+i;
            count++;
        }
        if (row == m*n-1) // last row
        {
            Lt_row_ptr[row+1] = count;
        }
    }
}

/**
 * @brief Compute Boundary Condition terms
*/
__global__ void BoundaryCondition_kernel(double* f, int m, int n, double h)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = i*n+j;

    if (i < m && j < n)
    {
        f[index] = 0;
        if (i == 0) // top row
        {
            f[index] += a(h*(j+1));
        }
        if (j == 0) // left column
        {
            f[index] += b(h*(i+1));
        }
        if (j == n-1) // right column
        {
            f[index] += d(h*(i+1));
        }
        if (i == m-1) // bottom row
        {
            f[index] += c(h*(j+1));
        }
    }
    
    
}

/**
 * @brief Backward Euler update b kernel
*/
__global__ void Updateb_kernel(double *b, double *u, double*f, double tinvhsq, int MATRIX_DIM)
{
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (id < MATRIX_DIM)
    {
        b[id] = u[id] + f[id]*tinvhsq;
    }
}