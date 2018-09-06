#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

#include "../headers/utils.h"
#include "../headers/training.h"

// Updates Z[0 or 1][layer][rows * columns]
// We specify rows as 1 when we do stochastic gd
void feedforward_update(double ***Z, int rows, int columns_Y, int columns_X, int layers,
                        double *X, double ***wb,
                        int nodes[layers], char funcs[layers+1][30])
{
  // l is for layers
  // i for each row
  // j for columns at each layer
  int l, for_helper = rows * nodes[0], i = for_helper - 1, j = nodes[0] - 1;
  
  // Input layer
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              rows, // Rows of z[0][l][0][j]
              nodes[0], // Columns of z[0][l][0][j]
              columns_X, // columns of A, rows of B
              1.0, // scaling factor (none)
              X, columns_X, // C = A * B -> matrix A, ldA
              wb[0][0], nodes[0], // C = A * B -> matrix B, ldB
              1.0, // scaling factor for C (none)
              Z[0][0], nodes[0]); // C, ldC
  
  while (i >= 0) {
    if (j < 0) {
      j = nodes[0] - 1;
    }
    Z[0][0][i--] += wb[1][0][j--];
  }
  activate(Z[1][0], Z[0][0], for_helper, funcs[0]);
  
  // Intermediate layers
  switch (layers > 1) {
    case 0:
      break;
    default:
      l = layers - 1;
      while (l >= 1) {
        for_helper = rows * nodes[l];
        i = for_helper - 1;
        j = nodes[l] - 1;
        
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    rows, // Rows of z[0][l][i][j]
                    nodes[l], // Columns of z[0][l][i][j]
                    nodes[l-1], // columns of A, rows of B
                    1.0, // scaling factor (none)
                    Z[1][l-1], nodes[l-1], // C = A * B -> matrix A, ldA
                    wb[0][l], nodes[l], // C = A * B -> matrix B, ldB
                    1.0, // scaling factor for C (none)
                    Z[0][l], nodes[l]); // C, ldC
        
        while (i >= 0) {
          if (j < 0) {
            j = nodes[l] - 1;
          }
          Z[0][l][i--] += wb[1][l][j--];
        }
        activate(Z[1][l], Z[0][l], for_helper, funcs[l]);
        l--;
      }
      break;
  }
  
  // Output layer
  for_helper = rows * columns_Y;
  i = for_helper - 1;
  j = columns_Y - 1;
  
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              rows, // Rows of z[0][l][i][j]
              columns_Y, // Columns of z[0][l][0][j]
              nodes[layers-1], // columns of A, rows of B
              1.0, // scaling factor (none)
              Z[1][layers-1], nodes[layers-1], // C = A * B -> matrix A, ldA
              wb[0][layers], columns_Y, // C = A * B -> matrix B, ldB
              1.0, // scaling factor for C (none)
              Z[0][layers], columns_Y); // C, ldC
  
  while (i >= 0) {
    if (j < 0) {
      j = columns_Y - 1;
    }
    Z[0][layers][i--] += wb[1][layers][j];
  }
  activate(Z[1][layers], Z[0][layers], for_helper, funcs[layers]);
}
