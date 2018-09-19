#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

#include "../headers/utils.h"

// Updates Z[0 or 1][layer][rows * columns]
// We specify rows as 1 when we do stochastic gd
void cpu_feedforward_update(const size_t rows, const size_t columns_Y, const size_t columns_X, const int layers,
                            double ***Z,
                            double *X, double ***wb,
                            const int nodes[layers], char funcs[layers+1][30])
{
  // l is for layers
  // i for each row
  // j for columns at each layer
  int l = 0, i, j, cols, for_helper;
  
  do {
    if (l > 0 && l < layers) {
      for_helper = (int)rows * nodes[l];
      cols = nodes[l];
      
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  rows, // Rows of z[0][l][i][j]
                  nodes[l], // Columns of z[0][l][i][j]
                  nodes[l-1], // columns of A, rows of B
                  1.0, // scaling factor (none)
                  Z[1][l-1], nodes[l-1], // C = A * B -> matrix A, ldA
                  wb[0][l], nodes[l], // C = A * B -> matrix B, ldB
                  0.0, // scaling factor for C (none)
                  Z[0][l], nodes[l]); // C, ldC
      
    } else if (l == 0) {
      for_helper = (int)rows * nodes[l];
      cols = nodes[l];
      
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  rows, // Rows of z[0][l][0][j]
                  nodes[l], // Columns of z[0][l][0][j]
                  columns_X, // columns of A, rows of B
                  1.0, // scaling factor (none)
                  X, columns_X, // C = A * B -> matrix A, ldA
                  wb[0][l], nodes[l], // C = A * B -> matrix B, ldB
                  0.0, // scaling factor for C (none)
                  Z[0][l], nodes[l]); // C, ldC
      
    } else {
      for_helper = rows * columns_Y;
      cols = columns_Y;
      
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  rows, // Rows of z[0][l][i][j]
                  columns_Y, // Columns of z[0][l][0][j]
                  nodes[l-1], // columns of A, rows of B
                  1.0, // scaling factor (none)
                  Z[1][l-1], nodes[l-1], // C = A * B -> matrix A, ldA
                  wb[0][l], columns_Y, // C = A * B -> matrix B, ldB
                  0.0, // scaling factor for C (none)
                  Z[0][l], columns_Y); // C, ldC
      
    }
    j = cols - 1;
    i = for_helper - 1;
    do {
      Z[0][l][i--] += wb[1][l][j--];
      if (j < 0) {
        j = cols - 1;
      }
    } while (i >= 0);
    
    activate(Z[1][l], Z[0][l], for_helper, funcs[l]);
    l++;
  } while (l < layers + 1);
}
