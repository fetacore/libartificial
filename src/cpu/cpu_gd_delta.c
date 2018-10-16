#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../../openblas/cblas.h"

#include "../headers/utils.h"

#define KRED  "\x1B[31m"
#define RESET "\033[0m"

// DELTAS FOR EVERY TYPE OF GRADIENT DESCENT
// (batch => rows = all rows)
// (stochastic => rows = 1)
// (mini-batch => rows = rows of mini set)

void cpu_gd_delta(double **restrict deltas,
                  const int *restrict rows, const int *restrict columns_Y, const int *restrict layers,
                  double *Y, double ***Z, double **w,
                  const int *restrict nodes, const int *restrict funcs)
{
  int l, i = (*rows) * (*columns_Y) - 1;
  
  //////////////////////////////////////////////////////////////////
  // Gradient of layer's unactivated output
  double **help_1 = malloc((*layers) * sizeof(double *));
  // Product of next layer's transposed weights and deltas
  double **help_2 = malloc((*layers) * sizeof(double *));
  // We do not need them at the output layer
  //////////////////////////////////////////////////////////////////
  if (help_1 && help_2) {
    for (l = 0; l < (*layers); l++) {
      help_1[l] = malloc((*rows) * nodes[l] * sizeof(double));
      help_2[l] = malloc((*rows) * nodes[l] * sizeof(double));
      if (help_1[l] && help_2[l]) {
        memset(help_1[l], 0.0, (*rows) * nodes[l] * sizeof(double));
        memset(help_2[l], 0.0, (*rows) * nodes[l] * sizeof(double));
      } else {
        printf(KRED "\nFailed to allocate helpers for delta. Aborting...\n" RESET);
        free(help_1);
        free(help_2);
        return;
      }
    }
  } else {
    printf(KRED "\nFailed to allocate helpers for delta. Aborting...\n" RESET);
    return;
  }
  
  // Last layer
  switch (funcs[(*layers)]) {
    // Linear case
    case 2:
      do {
        deltas[(*layers)][i] = Z[1][(*layers)][i] - Y[i];
      } while (--i >= 0);
      break;
    default:
      gradient(deltas[(*layers)], Z[0][(*layers)], rows, columns_Y, &funcs[(*layers)]);
      do {
        deltas[(*layers)][i] *= Y[i] - Z[1][(*layers)][i];
      } while (--i >= 0);
      break;
  }
  
  // Before last
  l = (*layers) - 1;
  i = (*rows) * nodes[l] - 1;
    
  gradient(help_1[l], Z[0][l], rows, &nodes[l], &funcs[l]);
  
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              (*rows), // Rows of help_2[0][j]
              nodes[l], // Columns of help_2[0][j]
              (*columns_Y), // columns of A, rows of B
              1.0, // scaling factor (none)
              deltas[l+1], (*columns_Y), // C = A * B -> matrix A, ldA
              w[l+1], (*columns_Y), // C = A * B -> matrix B, ldB
              0.0, // scaling factor for C (none)
              help_2[l], nodes[l]); // C, ldC
  
  // Hadamard product
  do {
    deltas[l][i] = help_1[l][i] * help_2[l][i];
  } while (--i >= 0);
  
  switch ((*layers) == 1) {
    case 1:
      for (l = 0; l < (*layers); l++) {
        free(help_2[l]);
        free(help_1[l]);
      }
      free(help_2);
      free(help_1);
      return;
    default:
      // All other layers
      l = (*layers) - 2;
      do {
        i = (*rows) * nodes[l] - 1;
        
        gradient(help_1[l], Z[0][l], rows, &nodes[l], &funcs[l]);
        
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (*rows), // Rows of help_2[0][j]
                    nodes[l], // Columns of help_2[0][j]
                    nodes[l+1], // columns of A, rows of B
                    1.0, // scaling factor (none)
                    deltas[l+1], nodes[l+1], // C = A * B -> matrix A, ldA
                    w[l+1], nodes[l+1], // C = A * B -> matrix B,  ldB
                    0.0, // scaling factor for C (none)
                    help_2[l], nodes[l]); // C, ldC
        
        // Hadamard product
        do {
          deltas[l][i] = help_1[l][i] * help_2[l][i];
        } while (--i >= 0);
      } while (--l >= 0);
      for (l = 0; l < (*layers); l++) {
        free(help_2[l]);
        free(help_1[l]);
      }
      free(help_2);
      free(help_1);
      return;
  }
}
