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
};


double get_ij(CSRMatrix& A, int i, int j) const 
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

void set_ij(CSRMatrix& A, int i, int j, double value) 
{
    // Find the range of non-zero elements in the row
    int start = A.row_ptr[i];
    int end = A.row_ptr[i + 1];

    // Check if the element already exists in the matrix
    for (int k = start; k < end; k++) {
        if (A.columns[k] == j) {
            A.values[k] = value;  // Update existing value
            return;
        }
    }
    return;
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
__global__ void init_L_Lt_cu(CSRMatrix& L, CSRMatrix& Lt, int m, int n){
    // Initialize L and Lt matrix 
    // to avoid too many shifting, we pre-allocate the non-zero terms with zero
    L.rows = A.rows; L.non_zeros = 0;
    Lt.rows = A.rows; Lt.non_zeros = 0;

    // L: row 0 only have 1 element
    L.row_ptr[0] = 0; int count = 0;
    // L: row i from 1 to n has i non-zero elements
    for (int i = 1; i < n+1; i++)
    {
        L.row_ptr[i] = L.row_ptr[i-1] + i; 
        for (int j = 0; j < i; j++)
        {
            L.columns[count] = j; L.values[count] = 0; count++;
        }
    }
    // L: row i from n+1 to m*n has n + 1 non-zero elements
    for (int i = n+1; i < m*n+1; i++)
    {
        L.row_ptr[i] = L.row_ptr[i-1] + n + 1; 
        for (int j = i - n - 1; j < i; j++)
        {
            L.columns[count] = j; L.values[count] = 0; count++;
        }
    }
    L.non_zeros = count;
    
    // Lt: row i from 0 to m-n have n + 1 non-zero elements
    count = 0; L.row_ptr[0] = 0;
    for (int i = 1; i < m*n-n+1; i++)
    {
        Lt.row_ptr[i] = Lt.row_ptr[i-1] + n + 1; 
        for (int j = -1; j < n; j++)
        {
            Lt.columns[count] = i+j; Lt.values[count] = 0; count++;
        }
    }
    // Lt: row i from m-n to m have m-i non-zero element
    for (int i = m*n-n; i < m*n; i++)
    {
        Lt.row_ptr[i+1] = Lt.row_ptr[i] + m*n - i; 
        for (int j = i; j < m*n; j++)
        {
            Lt.columns[count] = j; Lt.values[count] = 0; count++;
        }
    }
    Lt.non_zeros = count;
}

__global__ void ldlt_cholesky_decomposition_cu(CSRMatrix& A, CSRMatrix& L, CSRMatrix& Lt, double* D, int m, int n)
{
    // M : kernel
    // Backward Euler: A = I + M*t*invhsq 
    // so Ajj = 1 + 4*t*invhsq is fixed value
    
    // Assume L and Lt have zeros pre-allocated
    // init_L_Lt_cu(CSRMatrix& L, CSRMatrix& Lt, int m, int n);
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // LDL Cholesky Decomposition: we know L non-zero term position now   
    for (int j = 0; j < A.rows; j++) {
        double sumL = 0.0;
        // Calculate the diagonal element of L and update D
        for (int k = 0; k < j; k++) {
            double Ljk = L.get_ij(j, k);
            sumL += Ljk * Ljk * D[k];
        }
        D[j] = A.get_ij(j,j) - sumL;  
        L.set_ij(j,j,1);
        Lt.set_ij(j,j,1);

        // Calculate the off-diagonal elements of L on Row i, for all i > j, there are at most n+1 rows to update
        for (int i = j+1; i < j+n+1; i++) {
            double sumL2 = 0.0; 

            double Aij = A.get_ij(i, j);
            // sum up previous L values, find whether Lik or Ljk is zero or non-zero, there are at most n+1 non zero elements
            // TBD: unroll first n loops
            int st = std::max(0, j-n);
            for (int k = st; k < j; k++)
            {
                // \sum_{k=1}^{j-1} L_{ik}*L_{jk}*D_{k}
                sumL2 += L.get_ij(i,k) * L.get_ij(j,k) * D[k];
            }
            // store Lij if it is non-zeros, assume D[j] != 0
            if (Aij - sumL2 != 0) {
                double Lij = (Aij - sumL2) / D[j];
                L.set_ij(i,j,Lij); 
                Lt.set_ij(j,i,Lij);
            }
        }
        // finished row
        // std::cout << "finished col: " << j << std::endl;
    }
}