# 2d Heat equation FD solver on GPU

<!-- For the PDE:
$$
\begin{aligned}
    u_t(x;t) &= F(x, t, u(x;t), \Delta u(x;t)) \\
\end{aligned}
$$

Discretize $F$ and the forward Euler method is given by:

$$
\begin{aligned}
    \frac{U_{n}^{k+1} - U_{n}^{k+1}}{\tau} &= F(x, t, u(x;t), \Delta u(x;t)) \\
\end{aligned}
$$ -->

For heat equation $u_t(x;t) = F(x, t, u(x;t),  \Delta u) = \alpha \Delta u$, the 5-point stencil discretize $F$ with second-order spatial LTE:

$$
\begin{aligned}
    F_d = \Delta U_{m,n} = \frac{1}{h}\left(U_{m-1,n} + U_{m+1,n} + U_{m,n-1} + U_{m,n+1} - 4U_{m,n}\right)\\
\end{aligned}
$$

## FTCS (forward time-centered space) scheme

FTCS (forward time-centered space) scheme derives forward Euler method:

## Implementation

Flattening the 2D heat map $A_{mxn}$ into $(mn)$-dimensional vector:

$$
\vec x := [A_{11}, \cdots, A_{1n},
 A_{21}, \cdots A_{2n},
 \cdots A_{mn}]
 $$

Matrix for 5-point Laplacian:

size of the matrix = $m^2*n^2$

for the row $i * n + j$ which is corresponding to element $A_{ij}$:

$$
F_{in + j} = 
\left[
\begin{array}{cccccc}
    \cdots(0)\cdots & -1 & \cdots(0_n)\cdots & -1 & 4 & -1 & \cdots(0_n)\cdots & -1 & \cdots(0)\cdots\\
\end{array}
\right]
$$

### Solving linear system $A\vec x = f$

We first enforce $F$ to be a square matrix $(m = n)$

$F$ is diagonally dominant so is positive definite; it is symmetric so spd

$F$ is symmetric positive definite matrix, i.e. all its eigenvalues are real and positive

Method 1: using Cholesky Decomposition

### Boundary conditions

For 2D heat equation with Dirichlet boundary condition on ([0, X] x [0, Y]):

    u((x, 0), t) = a(x, t)
    u((x, Y), t) = d(x, t)
    u((0, y), t) = b(y, t)
    u((X, y), t) = c(y, t)

The boundary functions agree at the corners:

    a = b at (0,0)
    d = b at (0,Y)
    a = c at (X,0)
    d = c at (X,Y)

To match boundary conditions:

* For nodes on the boundary:
  * $U_{0j}$
  * $U_{i0}$
  * $U_{mn}$, $U_{m0}$, $U_{0n}$, $U_{00}$

### ldlt Cholesky Decomposition

Decompose spd matrix $A$ to into a product of matrices

$$
A = LDL^T
$$

The algorithm for ldlt Cholesky Decomposition:

$$
\begin{aligned}
D_{j} &= A_{jj} - \sum_{k=1}^{j-1} L_{jk}^2 D_k \\
L_{ij} &= \frac1{D_j}(A_{ij} - \sum_{k=1}^{j-1} L_{ik}L_{jk} D_k) \text{, for } i > j
\end{aligned}
$$
