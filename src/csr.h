#include <iostream>

// i-th row length: rowInd[i+1] - rowInd[i]
// i-th row access: for (i = rowInd[i]; i < rowInd[i+1]; i++) V[i]
// access element ij: V[rowInd[i] + j]; j must leq rowInd[i+1]-rowInd[i]
// it is hard to visit column wise

// CSR Storage
struct CSRMatrix {
    double* values;   // Array storing non-zero values of the matrix
    int* columns;     // Array storing column indices of non-zero values
    int* row_ptr;      // Array storing row pointers (indices where each row starts)
    int rows;          // Number of rows in the matrix
    int capacity;
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

/**
 * @brief This initialize the 5-point laplacian
 * 
 * @param A 
 * @param m 
 * @param n 
 */
void initializeCSRMatrix(CSRMatrix& A, int m, int n) {
    int matrix_size = m * n;
    int num_elements = m * n * m * n;

    // Initialize CSR matrix values and structure
    int index_curr, index_next, index_prev, index_up, index_down;
    int current_element = 0;

    for (int i = 0; i < matrix_size; ++i) {
        index_curr = i * matrix_size + i;

        // A_{i-1,j}
        index_up = index_curr - n;
        if (index_up >= i * matrix_size) {
            A.values[current_element] = -1.0;
            A.columns[current_element] = index_up;
            A.non_zeros++;
            current_element++;
        }

        // A_{i,j-1}
        index_prev = index_curr - 1;
        if (index_prev >= i * matrix_size) {
            A.values[current_element] = -1.0;
            A.columns[current_element] = index_prev;
            A.non_zeros++;
            current_element++;
        }

        // A_{i,j}
        A.values[current_element] = 4.0;
        A.columns[current_element] = index_curr;
        A.non_zeros++;
        current_element++;

        // A_{i,j+1}
        index_next = index_curr + 1;
        if (index_next < (i + 1) * matrix_size) {
            A.values[current_element] = -1.0;
            A.columns[current_element] = index_next;
            A.non_zeros++;
            current_element++;
        }
        
        // A_{i+1,j}
        index_down = index_curr + n;
        if (index_down < (i + 1) * matrix_size) {
            A.values[current_element] = -1.0;
            A.columns[current_element] = index_down;
            A.non_zeros++;
            current_element++;
        }

        // March the row pointer by 1
        A.row_ptr[i + 1] = current_element;
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
            if (valueIndex < A.row_ptr[i + 1] && A.columns[valueIndex] == i * rows + j) {
                std::cout << A.values[valueIndex] << " ";
                ++valueIndex;
            } else {
                std::cout << "0 ";
            }
        }
        std::cout << std::endl;
    }
}


/**
 * @brief Print out a CSRMatrix's values, rows, and columns
 * 
 * @param A 
 */
void print_csr_matrix_info(const CSRMatrix& A) {
    // Print values
    std::cout << "Values: ";
    for (int i = 0; i < A.non_zeros; ++i) {
        std::cout << A.values[i] << " ";
    }
    std::cout << std::endl;

    // Print row pointers
    std::cout << "Row Pointers: ";
    for (int i = 0; i <= A.rows; ++i) {
        std::cout << A.row_ptr[i] << " ";
    }
    std::cout << std::endl;

    // Print columns
    std::cout << "Columns: ";
    for (int i = 0; i < A.non_zeros; ++i) {
        std::cout << A.columns[i] << " ";
    }
    std::cout << std::endl;
}

/**
 * @brief Decompose spd matrix A into LDL,
 * 
 * @param A spd matrix in CSR Format
 * @param L Lower Triangular matrix in CSR Format
 * @param D diagonal matrix stored as a vector
 */
void ldlt_cholesky_decomposition_seq(const CSRMatrix& A, CSRMatrix& L, double* D) {
    int n = A.rows;

    // Initialize L and D matrices
    L.values = new double[A.capacity];
    L.columns = new int[A.capacity];
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
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) {
            int j = A.columns[p];
            double Aij = A.values[p];

            if (j > i) {
                double Lij = (Aij - sumL) / D[i];
                L.values[L.row_ptr[i + 1]] = Lij;
                L.columns[L.row_ptr[i + 1]] = j;
                L.row_ptr[i + 1]++;
            }
        }
    }
}
