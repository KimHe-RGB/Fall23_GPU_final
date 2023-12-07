/**
 * @file main.cpp
 * @author Xujin He (xh1131@nyu.edu)
 * @brief main function for solving 2D heat equation, sequential code implementation
 * @version 0.1
 * @date 2023-11-29
 * 
 * @copyright Copyright (c) 2023
 */

#include "linalg.h"
#include "load_heat_map.h"

const double h = 0.01;
const double invhsq = 1/h/h;
const double tau = 0.01; // timestep size ht
const double endT = 1; // end time

// boundary condtions on top
double a(double x)
{
    return 80;
}
// boundary condtions on left
double b(double y)
{
    return 40;
}
// boundary condtions on bottom
double c(double y)
{
    return 40;
}
// boundary conditions on right
double d(double x)
{
    return 80;
}

/**
 * @brief Use Backward Euler method to solve heat equation
 * For each time step, we solve the linear system (I+ht*A) \ (u+ht*f);
 * 
 * @param f 
 * @param u 
 * @param D 
 * @param uf 
 * @param DIM_X 
 * @param DIM_Y 
 */
void Backward_Euler(double *f, double *u, double* D, double* uf, const int DIM_X, const int DIM_Y)
{
    // (I+ht*A) \ (u+ht*f);
    double t = 0.0;
    const int MATRIX_DIM = DIM_X*DIM_Y;

    // left side
    CSRMatrix A = CSRMatrix(MATRIX_DIM, 5*DIM_X*DIM_Y);
    initBackwardEulerMatrix(A, tau*invhsq, DIM_X, DIM_Y); // initialize I+ht*invhsq*A
    CSRMatrix L = CSRMatrix(MATRIX_DIM, 5*DIM_X*DIM_X*DIM_Y);
    CSRMatrix Lt = CSRMatrix(MATRIX_DIM, 5*DIM_X*DIM_X*DIM_Y);
    // double D[MATRIX_DIM];
    ldlt_cholesky_decomposition_seq(A, L, Lt, D);
    // right side
    // double uf[MATRIX_DIM];
    double* u_temp;
    
    while(t < endT)
    {
        for (int i = 0; i < DIM_X; i++)
        {
            for (int j = 0; j < DIM_Y; j++)
            {
                uf[i*DIM_Y+j] = u[i*DIM_Y+j] + tau*invhsq*f[i*DIM_Y+j];
            }
        }
        solveAxb(L, Lt, D, uf, u, u_temp, MATRIX_DIM);
        t += tau;
    }
}

/**
 * @brief Given A = LDL', solve LDL'x = b
 * 
 */
void solveAxb(CSRMatrix &L, CSRMatrix &Lt, double *D, double *b, double *x, double *x_temp,  const int MATRIX_DIM)
{
    forward_substitute(b, L, x_temp);
    elementwise_division_vector(D, x_temp, MATRIX_DIM);
    backward_substitute(x_temp, Lt, x);
}

/**
 * @brief Compute the boundary condition term for the RHS, here we assume boundary condition to be constant 0 all the time
 * 
 * @param f the boundary condition 
 * @param m 
 */
void computeBoundaryCondition(double* f, double *u, const int m, const int n)
{
    f[0] = a(h) + b(h); // TL corner
    f[n-1] = b(n*h) + d(h); // TR corner
    f[(m-1)*n] = a(m*h) + c(h); // BL corner
    f[m*n-1] = c(m*h) + d(n*h); // BR corner
    // boundary condition on top row, except the 1st and nth col
    for (int i = 1; i < n-1; i++)
    {
        f[i] = b(h*(i+1));
    }
    // boundary condition on left and right col, except 1st and last row
    for (int i = n; i < (m-1)*n; i+=n)
    {
        f[i] = a(h*i/n);
        f[i+n-1] = d(h*i/n);
    }
    // boundary condition on bottom row, except the 1st and nth col
    for (int i = (m-1)*n + 1; i < m*n - 1; i++)
    {
        f[i] = c(h*(i-(m-1)*n));
    }
}

int main(int argc, char const *argv[])
{   
    float* Data; // 2D heat map vector = (m*n)
    const int DIM_X = 3; // grid dim = m
    const int DIM_Y = 3; // grid dim = n
    int DATA_SIZE = DIM_X * DIM_Y * sizeof(float);

    const int MATRIX_DIM = DIM_X * DIM_Y; // A = (mn x mn), this is extremely large

    // malloc CSR Matrix A in CPU
    CSRMatrix A = CSRMatrix(MATRIX_DIM, 5*DIM_X*DIM_Y);
    CSRMatrix L = CSRMatrix(MATRIX_DIM, 5*DIM_X*DIM_X*DIM_Y);
    CSRMatrix Lt = CSRMatrix(MATRIX_DIM, 5*DIM_X*DIM_X*DIM_Y);
    double D[MATRIX_DIM];

    initializeCSRMatrix(A, DIM_X, DIM_Y);
    // Test: init
    // print_csr_matrix_info(A);
    // print_csr_matrix(A);

    ldlt_cholesky_decomposition_seq(A, L, Lt, D);
    // Test: Cholesky works
    // print_csr_matrix_info(L);
    // print_csr_matrix(L);
    // print_csr_matrix_info(Lt);
    // print_csr_matrix(Lt);
    // print_diagonal(D, MATRIX_DIM);

    // Test: Solving Ax = b by Cholesky
    double b[] = {0.8147, 0.9058, 0.1270, 0.9134, 0.6324, 0.0975, 0.2785, 0.5469, 0.9575};
    double x[MATRIX_DIM]; 
    double *x_temp;
    solveAxb(L, Lt, D, b, x, x_temp, MATRIX_DIM);
    print_diagonal(x_temp, MATRIX_DIM);
    print_diagonal(x_temp, MATRIX_DIM);
    print_diagonal(x, MATRIX_DIM);


    // Test: load heat map from csv:
    // example initial condition is 76 x 76
    double u[76*76];
    loadCSV("../heat_map.csv", u, 76*76);
    print_diagonal(u, 76*76);
    writeCSV("../heat_map_out.csv", u, 76, 76);
    

    // initialize boundary condition correction term
    double f[76];
    computeBoundaryCondition(f, u, DIM_X, DIM_Y);
    // Backward Euler steps
    // Backward_Euler(A, tau, f, u, 76, 76);
    return 0;
}
