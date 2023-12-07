typedef CSRMatrix;

#include <cuda.h>

/**
 * @brief cuda kernel to compute forward substitution
 */
__global__ void forward_substitute_cu(const double* b, const CSRMatrix& L, double* x);
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
__global__ void backward_substitute_cu(const double* b, const CSRMatrix& U, double* x);
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