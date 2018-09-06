#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

#include "../headers/utils.h"
#include "../headers/training.h"
#include "../headers/prints.h"

double ***feedforward(int rows, int columns_Y, int columns_X, int layers, double *X, double ***wb,
                      int nodes[layers], char funcs[layers+1][30])
{
  // l is for layers
  // i is for each row * column of X, Y
  int l = layers, i = rows * columns_Y - 1;
  
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
  feedforward_update(Z, rows, columns_Y, columns_X, layers, X,wb, nodes, funcs);
  
  return Z;
}
