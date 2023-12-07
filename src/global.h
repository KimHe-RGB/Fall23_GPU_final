#ifndef __GLOBAL_H
#define __GLOBAL_H
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

#endif