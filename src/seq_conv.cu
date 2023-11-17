#include <time.h>
#include <cuda.h>

/**
 * @brief 
 * 
 * 
 * Convolution is only for explicit one-step time stepping, in which it is not required to solve ; 
 * 
 */
// h x h step size
const float h = 0.01;
const float invhsq = 1/h/h;

// 3 x 3 kernels -- 
// 5-point stencil
const float FIVE_POINT_LAPLACIAN[9] = 
{
    0,        1*invhsq,  0,
    1*invhsq, -4*invhsq, 1*invhsq, 
    0,        1*invhsq,  0
};
// Oono-Puri
const float NINE_POINT_LAPLACIAN_1[9] = 
{
    0.25*invhsq,   0.5*invhsq, 0.25*invhsq,
    0.66*invhsq,    -3*invhsq, 0.66*invhsq, 
    0.25*invhsq,   0.5*invhsq, 0.25*invhsq
};
// Patra-Karttunen / Mehrstellen
const float NINE_POINT_LAPLACIAN_2[9] = 
{
    0.16*invhsq,  0.66*invhsq, 0.16*invhsq,
    0.66*invhsq, -3.33*invhsq, 0.66*invhsq, 
    0.16*invhsq,  0.66*invhsq, 0.16*invhsq
};

const float t_h = 0.01; // timestep

int main(int argc, char const *argv[])
{
        
    float** Data;
    const int DIM_X = 256;
    const int DIM_Y = 512;
    const int DATA_SIZE = DIM_X * DIM_Y * sizeof(float);

    const float KER[9] = FIVE_POINT_LAPLACIAN; // pick a kernel
    const int KER_R = 1; // 3 x 3 kernel radius = 1

    // randomize a 2d Heat Map
    srand(time(0));
    for (i = 0; i < DIM_X; i++)
    {   
        for (j = 0; j < DIM_Y; j++) 
        {
            Data[i*DIM + j] = rand();
        }
    }

    /*----------- SIMPLE SEQUENTIAL CONVOLUTION -------------*/
    for (int i = KER_R; i < DIM_X - KER_R; ++i)
    {
        for (int j = KER_R; j < DIM_Y - KER_R; ++j)
        {
            int sum = 0;
            for (int ki = -KER_R; ki <= KER_R; ++ki) 
            {
                for (int kj = -KER_R; kj <= KER_R; ++kj) 
                {
                    sum += Data[(i + ki)*DIM_X + j + kj] * KER[ki + KER_R][kj + KER_R];
                }
            }
            Data[i*DIM_X + j] = sum;
        }
    }

    /*---  ----*/
    


    return 0;
}
