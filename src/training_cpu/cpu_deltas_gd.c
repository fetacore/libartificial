#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>

#include "../headers/utils.h"

// DELTAS FOR EVERY TYPE OF GRADIENT DESCENT
// (batch => rows = all rows)
// (stochastic => rows = 1)
// (mini-batch => rows = rows of mini set)

void delta_gd(double **deltas, const size_t rows, const size_t columns_Y, const int layers,
              const double *Y, double ***Z, double ***wb,
              const int nodes[layers], char funcs[layers+1][30])
{
  int l, i;
  size_t for_helper = rows * columns_Y;
  // Deltas filling
  //////////////////////////////////////////////////////////////////
  // Gradient of layer's unactivated output
  double **help_1 = malloc(layers * sizeof(double *));
  // Product of next layer's transposed weights and deltas
  double **help_2 = malloc(layers * sizeof(double *));
  // We do not need them at the output layer
  //////////////////////////////////////////////////////////////////
  
  // Last layer
  switch (strcmp(funcs[layers], "linear")) {
    case 0:
      i = for_helper - 1;
      do {
        deltas[layers][i] = Z[1][layers][i] - Y[i];
        i--;
      } while (i >= 0);
      break;
    default:
      gradient(deltas[layers], Z[0][layers], for_helper, funcs[layers]);
      i = for_helper - 1;
      do {
        deltas[layers][i] = (Z[1][layers][i] - Y[i]) * deltas[layers][i];
        i--;
      } while (i >= 0);
      break;
  }
  
  l = layers - 1;
  // Layers backwards
  do {
    for_helper = rows * nodes[l];
    
    help_1[l] = malloc(for_helper * sizeof(double));
    help_2[l] = malloc(for_helper * sizeof(double));
    
    gradient(help_1[l], Z[0][l], for_helper, funcs[l]);
    i = for_helper;
    do {
      help_2[l][i--] = 0.0;
    } while (i >= 0);
    
    switch (l == layers - 1) {
      // True
      case 1:
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    rows, // Rows of help_2[0][j]
                    nodes[l], // Columns of help_2[0][j]
                    columns_Y, // columns of A, rows of B
                    1.0, // scaling factor (none)
                    deltas[l+1], columns_Y, // C = A * B -> matrix A, ldA
                    wb[0][l+1], columns_Y, // C = A * B -> matrix B, ldB
                    0.0, // scaling factor for C (none)
                    help_2[l], nodes[l]); // C, ldC
        
        // Hadamard product
        i = for_helper;
        do {
          deltas[l][i] = help_1[l][i] * help_2[l][i];
          i--;
        } while (i >= 0);
        
        free(help_2[l]);
        free(help_1[l]);
        l--;
        continue;
      // False
      default:
        // find help_2
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    rows, // Rows of help_2[0][j]
                    nodes[l], // Columns of help_2[0][j]
                    nodes[l+1], // columns of A, rows of B
                    1.0, // scaling factor (none)
                    deltas[l+1], nodes[l+1], // C = A * B -> matrix A, ldA
                    wb[0][l+1], nodes[l+1], // C = A * B -> matrix B,  ldB
                    0.0, // scaling factor for C (none)
                    help_2[l], nodes[l]); // C, ldC
        
        // Hadamard product
        i = for_helper;
        do {
          deltas[l][i] = help_1[l][i] * help_2[l][i];
          i--;
        } while (i >= 0);
        
        free(help_2[l]);
        free(help_1[l]);
        l--;
        continue;
    }
  } while (l >= 0);
  free(help_2);
  free(help_1);
}
