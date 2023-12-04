#include <iostream>

struct CSRMatrix {
    double* values;   // Array storing non-zero values of the matrix
    int* columns;     // Array storing column indices of non-zero values
    int* row_ptr;      // Array storing row pointers (indices where each row starts)
    int rows;          // Number of rows in the matrix
    int non_zeros;     // Number of non-zero elements in the matrix

    // Constructor to initialize CSRMatrix
    CSRMatrix(int rows, int non_zeros) : rows(rows), non_zeros(non_zeros) {
        values = new double[non_zeros];
        columns = new int[non_zeros];
        row_ptr = new int[rows + 1];
    }

    // Destructor to free allocated memory
    ~CSRMatrix() {
        delete[] values;
        delete[] columns;
        delete[] row_ptr;
    }
};

// Function to perform LDL Cholesky Decomposition on a CSR matrix
void ldltCholeskyDecomposition(const CSRMatrix& A, CSRMatrix& L, double* D) {
    int n = A.rows;

    // Initialize L and D matrices
    L.values = new double[A.non_zeros];
    L.columns = new int[A.non_zeros];
    L.row_ptr = new int[n + 1];

    // LDL Cholesky Decomposition
    for (int i = 0; i < n; ++i) {
        double sumL = 0.0;

        // Calculate the diagonal element of L and update D
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) {
            int j = A.columns[p];
            double Aij = A.values[p];

            if (j < i) {
                sumL += L.values[p] * L.values[p] * D[j];
            } else if (j == i) {
                D[i] = Aij - sumL;
                L.row_ptr[i + 1] = L.row_ptr[i] + 1;
                L.values[L.row_ptr[i]] = 1.0;  // Diagonal element of L is 1
                L.columns[L.row_ptr[i]] = i;
            }
        }

        // Calculate the off-diagonal elements of L
        for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
            int j = A.columns[k];
            double Aij = A.values[k];
            if (j > i) {
                double Lij = (Aij - sumL) / D[i]; 
                L.values[L.non_zeros] = Lij;  
                L.columns[L.non_zeros] = j;
                L.row_ptr[i + 1]++;
                L.non_zeros++;
            }
        }

    }
}

/**
 * @brief Print out a CSRMatrix as a full matrix
 * 
 * @param A 
 */
void print_csr_matrix(const CSRMatrix& A) {
    int rows = A.rows;

    for (int i = 0; i < rows; ++i) {
        int valueIndex = A.row_ptr[i];
        for (int j = 0; j < rows; ++j) {
            if (valueIndex < A.row_ptr[i + 1] && A.columns[valueIndex] == j) {
                std::cout << A.values[valueIndex] << " ";
                ++valueIndex;
            } else {
                std::cout << " 0 ";
            }
        }
        std::cout << std::endl;
    }
}

int main() {
    // Example usage
    int rows = 3;
    int non_zeros = 6;
    CSRMatrix A(rows, non_zeros);
    double value[] = {4,-1,-1,4,-1,-1,4};
    int columns[] = {0,1,0,1,2,1,2};
    int row_ptr[] = {0,2,5,7};
    A.values = value;
    A.row_ptr = row_ptr;
    A.columns = columns;
    A.rows = 3;
    A.non_zeros = 6;
    print_csr_matrix(A);


    CSRMatrix L(rows, non_zeros);
    double* D = new double[rows];

    // Perform LDL Cholesky Decomposition
    ldltCholeskyDecomposition(A, L, D);

    // Print the results (L, D)
    print_csr_matrix(L);

    // Don't forget to free allocated memory
    delete[] D;
    delete[] L.values;
    delete[] L.columns;
    delete[] L.row_ptr;

    return 0;
}
