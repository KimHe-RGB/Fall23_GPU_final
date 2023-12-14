#ifndef __GLOBAL_H
#define __GLOBAL_H

#ifndef __IN_CUDA
const double h = 0.01;
const double invhsq = 1/h/h;
const double tau = 0.01; // timestep size ht
#else
const double h = 0.01;
const double invhsq = 1/h/h;
const double tau = 0.01; // timestep size ht
__device__ const double invhsq_dev = 10000;
__device__ const double tau_dev = 0.01; // timestep size ht
#endif

const double endT = 1; // end time

#define a(x) 3
#define b(y) 3
#define c(y) 10
#define d(x) 10 
// // boundary condtions on top
// double a(double x)
// {
//     return 3;
// }
// // boundary condtions on left
// double b(double y)
// {
//     return 3;
// }
// // boundary condtions on bottom
// double c(double y)
// {
//     return 10;
// }
// // boundary conditions on right
// double d(double x)
// {
//     return 10;
// }

#endif