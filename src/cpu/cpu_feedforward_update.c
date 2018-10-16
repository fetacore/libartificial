#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../openblas/cblas.h"

#include "../headers/utils.h"

// Updates Z[0 or 1][layer][rows * columns]
// We specify rows as 1 when we do stochastic gd
void cpu_feedforward_update(const int *restrict rows, const int *restrict columns_Y, const int *restrict columns_X, 
                            const int *restrict layers,
                            double ***restrict Z,
                            const double *restrict X, double **restrict w,
                            const int *restrict nodes, const int *restrict funcs)
{
  // l is for layers
  int l = 0;
  
  do {
    switch (l == 0) {
      case 1:
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (*rows), // Rows of z[0][l][0][j]
                    nodes[l], // Columns of z[0][l][0][j]
                    (*columns_X), // columns of A, rows of B
                    1.0, // scaling factor (none)
                    X, (*columns_X), // C = A * B -> matrix A, ldA
                    w[l], nodes[l], // C = A * B -> matrix B, ldB
                    0.0, // scaling factor for C (none)
                    Z[0][l], nodes[l]); // C, ldC
        activate(Z[1][l], Z[0][l], rows, &nodes[l], &funcs[l]);
        ++l;
        continue;
      default:
        break;
    }
    
    switch (l > 0 && l < (*layers)) {
      case 1:
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (*rows), // Rows of z[0][l][i][j]
                    nodes[l], // Columns of z[0][l][i][j]
                    nodes[l-1], // columns of A, rows of B
                    1.0, // scaling factor (none)
                    Z[1][l-1], nodes[l-1], // C = A * B -> matrix A, ldA
                    w[l], nodes[l], // C = A * B -> matrix B, ldB
                    0.0, // scaling factor for C (none)
                    Z[0][l], nodes[l]); // C, ldC
        activate(Z[1][l], Z[0][l], rows, &nodes[l], &funcs[l]);
        ++l;
        continue;
      default:
        break;
    }
    
    switch (l == (*layers)) {
      case 1:
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (*rows), // Rows of z[0][l][i][j]
                    (*columns_Y), // Columns of z[0][l][0][j]
                    nodes[l-1], // columns of A, rows of B
                    1.0, // scaling factor (none)
                    Z[1][l-1], nodes[l-1], // C = A * B -> matrix A, ldA
                    w[l], (*columns_Y), // C = A * B -> matrix B, ldB
                    0.0, // scaling factor for C (none)
                    Z[0][l], (*columns_Y)); // C, ldC
        activate(Z[1][l], Z[0][l], rows, columns_Y, &funcs[l]);
        return;
    }
  } while (l <= (*layers));
}
