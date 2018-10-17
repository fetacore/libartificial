#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../openblas/cblas.h"

#include "../headers/utils.h"
#include "../headers/cpu.h"

#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define RESET "\033[0m"

// GRADIENT DESCENT with batch size
// Directly manipulates weights and prediction

// For stochastic set batch = 1
// For mini-batch set batch = whatever you want (perfect divisor of rows)
// For pure batch gd set batch = rows

// returns new wb
void threaded_update(const double *restrict X, const double *restrict delta,
                     double *restrict grad_w,
                     double *restrict w,
                     const int *restrict m,
                     const int *restrict n,
                     const int *restrict k,
                     const double *restrict correction)
{  
  int i = (*m) * (*n) - 1;
  
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
              (*m), // Rows of grad_w[l]
              (*n), // Columns of grad_w[l]
              (*k), // columns of A, rows of B
              (*correction), // scaling factor
              X, (*m), // C = A * B -> matrix A, ldA
              delta, (*n), // C = A * B -> matrix B, ldB
              0.0, // scaling factor for C (none)
              grad_w, (*n)); // C, ldC
  
  do {
    w[i] -= grad_w[i];
  } while (--i >= 0);
}

void cpu_gd_train(const int rows, const int columns_Y, const int columns_X,
                  const int batch, const int layers, const int nodes[layers],
                  const double *Y, const double *X, register double **w,
                  char functions[layers + 1][30],
                  const double learning_rate, const int epochs)
{
  // l for layers
  // i for rows
  // j for columns
  // e for epochs
  int l, i, j, for_helper_w, for_helper_batch, e = epochs;
  register int *funcs;
  funcs = name2int(layers, functions);
    
  // Multiplication in threads
  openblas_set_num_threads(1);
  goto_set_num_threads(1);
  
  // In case of mini-batching or stochastic
  // b for batch
  // r helper for batch rows
  int b, r, r_over_b = rows/batch;
  
  double loss = 0.0;
  
  // For the averaging of deltas in batch/mini-batch
  double correction = 1.0;
  
  register double **deltas = malloc((layers + 1) * sizeof(double *));
  // The values to be subtracted from weights
  register double **grad_w = malloc((layers + 1) * sizeof(double *));
  //////////////////////////////////////////////////////////////////
  // Gradient of layer's unactivated output
  double **help_1 = malloc(layers * sizeof(double *));
  // Product of next layer's transposed weights and deltas
  double **help_2 = malloc(layers * sizeof(double *));
  // We do not need them at the output layer
  //////////////////////////////////////////////////////////////////
  if (deltas && grad_w && help_1 && help_2) {
    // Allocations
    for (l = layers + 1; l--; ) {
      if (l == 0) {
        for_helper_batch = batch * nodes[l];
        for_helper_w = columns_X * nodes[l];
        help_1[l] = malloc(batch * nodes[l] * sizeof(double));
        help_2[l] = malloc(batch * nodes[l] * sizeof(double));
        memset(help_1[l], 0.0, batch * nodes[l] * sizeof(double));
        memset(help_2[l], 0.0, batch * nodes[l] * sizeof(double));
      } else if (l == layers) {
        for_helper_batch = batch * columns_Y;
        for_helper_w = nodes[l-1] * columns_Y;
      } else {
        for_helper_batch = batch * nodes[l];
        for_helper_w = nodes[l-1] * nodes[l];
        help_1[l] = malloc(batch * nodes[l] * sizeof(double));
        help_2[l] = malloc(batch * nodes[l] * sizeof(double));
        memset(help_1[l], 0.0, batch * nodes[l] * sizeof(double));
        memset(help_2[l], 0.0, batch * nodes[l] * sizeof(double));
      }
      deltas[l] = malloc(for_helper_batch * sizeof(double));    
      grad_w[l] = malloc(for_helper_w * sizeof(double));
      if (deltas[l] && grad_w[l]) {
        memset(deltas[l], 0.0, for_helper_batch * sizeof(double));
        memset(grad_w[l], 0.0, for_helper_w * sizeof(double));
      } else {
        printf(KRED "\nFailed to allocate deltas, gradients or helpers. Aborting...\n" RESET);
        free(deltas);
        free(grad_w);
        free(help_2);
        free(help_1);
        return;
      }
    }
  } else {
    printf(KRED "\nFailed to allocate deltas, gradients or helpers. Aborting...\n" RESET);
    return;
  }
  
  register double ***Z = cpu_feedforward_cache(&rows, &columns_Y, &columns_X, &layers, X, w, nodes, funcs);
  
  // Big switch in case we have pure batch (do not allocate mini-batches)
  switch (batch == rows) {
    // True
    case 1:
      correction = learning_rate * 1.0/(double)rows;
      
      // Training
      do {
        // Find deltas
        cpu_gd_delta(deltas, help_1, help_2, &rows, &columns_Y, &layers, Y, Z, w, nodes, funcs);
        
        // Now we update the weights and biases
//         #pragma omp parallel for
        for (l = layers + 1; l--; ) {
          switch (l == 0) {
            case 1:
              threaded_update(X, deltas[l], grad_w[l], w[l], &columns_X, &nodes[l], &rows, &correction);
              continue;
            default:
              break;
          } 
          switch (l > 0 && l < layers) {
            case 1:
              threaded_update(Z[1][l-1], deltas[l], grad_w[l], w[l], &nodes[l-1], &nodes[l], &rows, &correction);
              continue;
            default:
              break;
          }
          switch (l == layers) {
            case 1:
              threaded_update(Z[1][l-1], deltas[l], grad_w[l], w[l], &nodes[l-1], &columns_Y, &rows, &correction);
              break;
            default:
              break;
          }
        }
        
        // Update Zs with the new wb's
        cpu_feedforward_update(&rows, &columns_Y, &columns_X, &layers, Z, X, w, nodes, funcs);
        
        switch (funcs[layers]) {
          // If softmax then cross entropy
          case 4:
            loss = xentropy(&rows, &columns_Y, Y, Z[1][layers]);
            break;
          default:
            loss = rmse(&rows, &columns_Y, Y, Z[1][layers]);
            break;
        }
        
        if (loss != loss) {
          printf(KRED "\nWe got NaN values during training. Aborting...\n" RESET);
          for (l = 0; l < layers + 1; l++) {
            free(deltas[l]);
            free(grad_w[l]);
            if (l < layers) {
              free(help_2[l]);
              free(help_1[l]);
            }
          }
          free(deltas);
          free(grad_w);
          free(funcs);
          free(help_2);
          free(help_1);
          delete_Z(layers, Z);
          return;
        }
        printf("\nLoss = %.10lf at epoch = %d\n", loss, epochs - e);
      } while (--e >= 0);
      break;
    default:
      correction = learning_rate * 1.0/(double)batch;
      
      // They need to be allocated once
      double **X_batch = malloc(r_over_b * sizeof(double *));
      double **Y_batch = malloc(r_over_b * sizeof(double *));
      
      if (X_batch && Y_batch) {
        for (i = r_over_b; i--; ) {
          X_batch[i] = malloc(batch * columns_X * sizeof(double));
          Y_batch[i] = malloc(batch * columns_Y * sizeof(double));
          if (X_batch[i] && Y_batch[i]) {
            memset(X_batch[i], 0.0, batch * columns_X * sizeof(double));
            memset(Y_batch[i], 0.0, batch * columns_Y * sizeof(double));
          } else {
            printf(KRED "\nFailed to allocate X or Y batches. Aborting...\n" RESET);
            free(X_batch);
            free(Y_batch);
            return;
          }
        }
      } else {
        printf(KRED "\nFailed to allocate X or Y batches. Aborting...\n" RESET);
        return;
      }
      
      i = r_over_b - 1;
      // Fill X_batch, Y_batch
      do {
        memcpy(X_batch[i], X + i * batch * columns_X, batch * columns_X * sizeof(double));
        memcpy(Y_batch[i], Y + i * batch * columns_Y, batch * columns_Y * sizeof(double));
      } while (--i >= 0);
      
      // The perceptrons' outputs
      double ***Z_batch = malloc(2 * sizeof(double **));
      if (Z_batch) {
        // Unactivated
        Z_batch[0] = malloc((layers + 1) * sizeof(double *));
        // Activated
        Z_batch[1] = malloc((layers + 1) * sizeof(double *));
        if (Z_batch[0] && Z_batch[1]) {
          for (l = layers + 1; l--; ) {
            switch (l == layers) {
              // Last layer
              case 1:
                Z_batch[0][l] = malloc(batch * columns_Y * sizeof(double));
                Z_batch[1][l] = malloc(batch * columns_Y * sizeof(double));
                if (Z_batch[0][l] && Z_batch[1][l]) {
                  memset(Z_batch[0][l], 0.0, batch * columns_Y * sizeof(double));
                  memset(Z_batch[1][l], 0.0, batch * columns_Y * sizeof(double));
                } else {
                  printf(KRED "\nFailed to allocate Z batches. Aborting...\n" RESET);
                  free(Z_batch[0]);
                  free(Z_batch[1]);
                  return;
                }
                continue;
              default:
                Z_batch[0][l] = malloc(batch * nodes[l] * sizeof(double));
                Z_batch[1][l] = malloc(batch * nodes[l] * sizeof(double));
                if (Z_batch[0][l] && Z_batch[1][l]) {
                  memset(Z_batch[0][l], 0.0, batch * nodes[l] * sizeof(double));
                  memset(Z_batch[1][l], 0.0, batch * nodes[l] * sizeof(double));
                } else {
                  printf(KRED "\nFailed to allocate Z batches. Aborting...\n" RESET);
                  free(Z_batch[0]);
                  free(Z_batch[1]);
                  return;
                }
                continue;
            }
          }
        } else {
          printf(KRED "\nFailed to allocate Z batches. Aborting...\n" RESET);
          free(Z_batch);
          return;
        }
      } else {
        printf(KRED "\nFailed to allocate Z batches. Aborting...\n" RESET);
        return;
      }
      
      do {
        i = r_over_b - 1;
        do {
          printf("     \t#%d/%d\n", i + 1, r_over_b);
          
          // Fill Z_batch with new Zs
          for (l = layers; l >= 0; l--) {
            switch (l == layers) {
              case 1:
                memcpy(Z_batch[0][l], Z[0][l] + i * batch * columns_Y, batch * columns_Y * sizeof(double));
                memcpy(Z_batch[1][l], Z[1][l] + i * batch * columns_Y, batch * columns_Y * sizeof(double));
                break;
              default:
                memcpy(Z_batch[0][l], Z[0][l] + i * batch * nodes[l], batch * nodes[l] * sizeof(double));
                memcpy(Z_batch[1][l], Z[1][l] + i * batch * nodes[l], batch * nodes[l] * sizeof(double));
                break;
            }
          }
          
          // Fill the deltas
          cpu_gd_delta(deltas, help_1, help_2, &batch, &columns_Y, &layers, Y_batch[i], Z_batch, w, nodes, funcs);
          
          // Now we update the weights and biases
//           #pragma omp parallel for
          for (l = layers + 1; l--; ) {
            switch (l == 0) {
              case 1:
                threaded_update(X, deltas[l], grad_w[l], w[l], &columns_X, &nodes[l], &batch, &correction);
                continue;
              default:
                break;
            }
            switch (l > 0 && l < layers) {
              case 1:
                threaded_update(Z[1][l-1], deltas[l], grad_w[l], w[l], &nodes[l-1], &nodes[l], &batch, &correction);
                continue;
              default:
                break;
            }
            switch (l == layers) {
              case 1:
                threaded_update(Z[1][l-1], deltas[l], grad_w[l], w[l], &nodes[l-1], &columns_Y, &batch, &correction);
                break;
              default:
                break;
            }
          }
          
          // Do an update of Z's with the new wb's
          cpu_feedforward_update(&rows, &columns_Y, &columns_X, &layers, Z, X, w, nodes, funcs);
        } while (--i >= 0);
        
        switch (funcs[layers]) {
          // If softmax then cross entropy
          case 4:
            loss = xentropy(&rows, &columns_Y, Y, Z[1][layers]);
            break;
          default:
            loss = rmse(&rows, &columns_Y, Y, Z[1][layers]);
            break;
        }
        
        if (loss != loss) {
          printf(KRED "\nWe got NaN values during training. Aborting...\n" RESET);
          // Free before quitting
          for (l = 0; l < layers + 1; l++) {
            free(Z_batch[0][l]);
            free(Z_batch[1][l]);
            free(deltas[l]);
            free(grad_w[l]);
            if (l < layers) {
              free(help_2[l]);
              free(help_1[l]);
            }
          }
          free(Z_batch[0]);
          free(Z_batch[1]);
          free(Z_batch);
          free(help_2);
          free(help_1);
          free(deltas);
          free(grad_w);
          
          for (i = 0; i < rows/batch; i++) {
            free(X_batch[i]);
            free(Y_batch[i]);
          }
          free(Y_batch);
          free(X_batch);
          
          free(funcs);
          delete_Z(layers, Z);
          return;
        }
        printf("\nLoss = %.10lf at epoch = %d\n", loss, epochs - e);
      } while (--e >= 0);

      // End of batching
      for (l = 0; l < layers + 1; l++) {
        free(Z_batch[0][l]);
        free(Z_batch[1][l]);
      }
      free(Z_batch[0]);
      free(Z_batch[1]);
      free(Z_batch);
      
      for (i = 0; i < r_over_b; i++) {
        free(X_batch[i]);
        free(Y_batch[i]);
      }
      free(Y_batch);
      free(X_batch);
      break;
  }
  
  // Save weights and free memory
  save_w(w, layers, nodes, columns_Y, columns_X);
  
  // Free the rest
  for (l = 0; l < layers + 1; l++) {
    free(deltas[l]);
    free(grad_w[l]);
    if (l < layers) {
      free(help_2[l]);
      free(help_1[l]);
    }
  }
  free(deltas);
  free(grad_w);
  free(funcs);
  free(help_2);
  free(help_1);
  delete_Z(layers, Z);
}
