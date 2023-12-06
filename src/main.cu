/**
 * @file main.cu
 * 
 * @author Xujin He (xh1131@nyu.edu)
 * @brief This is cuda code to run
 * @version 0.1
 * @date 2023-12-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <time.h>
#include <cuda.h>
#include <iostream>

const double h = 0.01;
const double invhsq = 1/h/h;
const double tau = 0.01; // timestep size


int main(int argc, char const *argv[])
{   
    float* Data; // 2D heat map vector = (m*n)
    const int DIM_X = 256; // grid dim = m
    const int DIM_Y = 256; // grid dim = n
    const int DATA_SIZE = DIM_X * DIM_Y * sizeof(float);

    const int MATRIX_DIM = DIM_X * DIM_X * DIM_Y * DIM_Y; // A = (mn x mn), this is extremely large so 

    // Get a 2d Heat Map

    // malloc CSR Matrix A in GPU


    return 0;
}