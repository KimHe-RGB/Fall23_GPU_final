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
#include "global.h"


int main(int argc, char const *argv[])
{   
    const int dim = 8;
    double u[dim*dim];
    loadCSV("../heat_map.csv", u, dim*dim);
    double d[dim*dim];
    double temp_vec[dim*dim];

    // Get a 2d Heat Map

    // malloc CSR Matrix A in GPU


    return 0;
}