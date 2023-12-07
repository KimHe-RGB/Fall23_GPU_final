#include <cuda.h>

// CSR Storage
struct CSRMatrix {
    double* values;   // Array storing non-zero values of the matrix
    int* columns;     // Array storing column indices of non-zero values
    int* row_ptr;      // Array storing row pointers (indices where each row starts)
    int rows;          // Number of rows in the matrix
    int capacity;      // maximum number of non-zeros
    int non_zeros;     // Number of non-zero elements in the matrix; Essentially "Capacity";

    // Constructor to initialize CSR Matrix
    CSRMatrix(int rows, int capacity) : rows(rows), capacity(capacity) {
        values = new double[capacity];
        columns = new int[capacity];
        row_ptr = new int[rows + 1];
        non_zeros = 0;
    }

    // Destructor to free allocated memory
    ~CSRMatrix() {
        delete[] values;
        delete[] columns;
        delete[] row_ptr;
    }

    double get_ij(int i, int j) const 
    {
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
            if (columns[p] == j) {
                return values[p];
            }
        }
        // Column not found, as columns are in ascending order
        return 0;
    }

    void set_ij(int i, int j, double value) 
    {
        // Find the range of non-zero elements in the row
        int start = row_ptr[i];
        int end = row_ptr[i + 1];

        // Check if the element already exists in the matrix
        for (int k = start; k < end; k++) {
            if (columns[k] == j) {
                values[k] = value;  // Update existing value
                return;
            }
        }

        // for large matrix, this branch is the most time-consuming step so we need to avoid
        // If the element doesn't exist, add it to the matrix
        // std::cout << "Bad Branch: undesired insertion at " << i << ", " << j << std::endl;
        int insertPos = end;
        for (int k = start; k < end; k++) {
            if (columns[k] > j) {
                insertPos = k;
                break;
            }
        }

        // Shift all existing non_zeros values and columns to make space for the new element
        for (int k = non_zeros; k > insertPos; k--) {
            values[k] = values[k - 1];
            columns[k] = columns[k - 1];
        }

        // Insert the new element
        values[insertPos] = value;
        columns[insertPos] = j;

        // Update row pointers after insertion
        for (int k = i + 1; k <= rows; k++) {
            row_ptr[k]++;
        }
        non_zeros++;

        // std::cout << i << ", " << j << " = " << values[insertPos] << std::endl;
        // std::cout << "Values: ";
        // for (int i = 0; i < non_zeros; ++i) {
        //     std::cout << values[i] << " ";
        // }
        // std::cout << std::endl;
    }
};

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

__global__ void init_L_cu(CSRMatrix& L, int m, int n)
{

}
__global__ void ldlt_cholesky_decomposition_cu(CSRMatrix& A, CSRMatrix& L, CSRMatrix& Lt, double* D, int m, int n)
{

}