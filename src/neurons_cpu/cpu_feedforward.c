#include <stdio.h>
#include <stdlib.h>

#include "../headers/utils.h"
#include "../headers/training_cpu.h"

double ***feedforward(const size_t rows, const size_t columns_Y, const size_t columns_X, const int layers,
                      const double *X, double ***wb,
                      const int nodes[layers], char funcs[layers+1][30])
{
  // l is for layers
  // i is for each row * column of X, Y
  int l = layers;
  int i = rows * columns_Y - 1;
    
  // feeds at every layer
  double ***Z = malloc(2 * sizeof(double **));
  Z[0] = malloc((layers + 1) * sizeof(double *));
  Z[1] = malloc((layers + 1) * sizeof(double *));
  
  Z[0][l] = malloc((i + 1) * sizeof(double));
  Z[1][l] = malloc((i + 1) * sizeof(double));
  while (i >= 0) {
    Z[0][l][i] = 0.0;
    Z[1][l][i--] = 0.0;
  }
  
  for (l = 0; l < layers; l++) {
    i = rows * nodes[l] - 1;
    Z[0][l] = malloc((i + 1) * sizeof(double));
    Z[1][l] = malloc((i + 1) * sizeof(double));
    while (i >= 0) {
      Z[0][l][i] = 0.0;
      Z[1][l][i--] = 0.0;
    }
  }
  
  // Directly manipulates Z
  feedforward_update(Z, rows, columns_Y, columns_X, layers, X, wb, nodes, funcs);  
  return Z;
}
