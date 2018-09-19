#include <stdio.h>
#include <stdlib.h>

#include "../headers/utils.h"
#include "../headers/training_cpu.h"

double ***cpu_feedforward(const size_t rows, const size_t columns_Y, const size_t columns_X, const int layers,
                          double *X, double ***wb,
                          const int nodes[layers], char funcs[layers+1][30])
{
  // l is for layers
  // i is for each row * column of X, Y
  int l = layers;
  int i = rows * columns_Y;
    
  // feeds at every layer
  double ***Z = calloc(2, sizeof(double **));
  Z[0] = calloc(layers + 1, sizeof(double *));
  Z[1] = calloc(layers + 1, sizeof(double *));
  
  Z[0][l] = calloc(i, sizeof(double));
  Z[1][l] = calloc(i, sizeof(double));
  
  for (l = 0; l < layers; l++) {
    i = (int)rows * nodes[l];
    Z[0][l] = calloc(i, sizeof(double));
    Z[1][l] = calloc(i, sizeof(double));
  }
  
  // Directly manipulates Z
  cpu_feedforward_update(rows, columns_Y, columns_X, layers, Z, X, wb, nodes, funcs);  
  return Z;
}
