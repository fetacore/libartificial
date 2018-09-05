#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

#include "../headers/utils.h"
#include "../headers/training.h"
#include "../headers/neurons.h"

// GRADIENT DESCENT with batch size
// Directly manipulates weights and prediction

// For stochastic set batch = 1
// For mini-batch set batch = whatever you want (perfect divisor of rows)
// For pure batch gd set batch = rows

void update_gd(int rows, int columns_Y, int columns_X, int batch, int layers, int nodes[layers],
               double *Y, double *X, double ***Z, double ***wb,
               char funcs[layers+1][30], double learning_rate, int epochs)
{
  // l for layers
  // i for rows
  // j for columns
  // e for epochs
  int l, i, j, e = epochs;
  
  // In case of mini-batching or stochastic
  // b for batch
  // r helper for batch rows
  int b, r, r_over_b = rows/batch, for_helper_w = columns_X * nodes[0];
  
  double loss = 0.0;
  
  // For the averaging of deltas in batch/mini-batch
  double correction = 1.0;
  
  // The values to be subtracted from weights
  double **grad_w = malloc((layers + 1) * sizeof(double *));
  
  double **deltas = malloc((layers + 1) * sizeof(double *));
  
  // For batch/mini-batch
  double **deltas_row_sum = malloc((layers + 1) * sizeof(double *));
  
  // Allocations
  
  // delta dim (batch * nodes[0])
  deltas[0] = malloc(batch * nodes[0] * sizeof(double));
  deltas_row_sum[0] = malloc(nodes[0] * sizeof(double));
  
  for (i = 0; i < batch * nodes[0]; i++) {
    deltas[0][i] = 0.0;
    switch (i < nodes[0]) {
      case 1:
        deltas_row_sum[0][i] = 0.0;
        break;
      default:
        break;
    }
  }
  
  grad_w[0] = malloc(for_helper_w * sizeof(double));
  for (i = 0; i < for_helper_w; i++) {
    grad_w[0][i] = 0.0;
  }
  
  switch (layers > 1) {
    // More than one hidden layers
    case 1:
      for (l = 1; l < layers; l++) {
        // delta dim (batch * nodes[l])
        deltas[l] = malloc(batch * nodes[l] * sizeof(double));
        deltas_row_sum[l] = malloc(nodes[l] * sizeof(double));
        
        for (i = 0; i < batch * nodes[l]; i++) {
          deltas[l][i] = 0.0;
          switch (i < nodes[l]) {
            case 1:
              deltas_row_sum[l][i] = 0.0;
              break;
            default:
              break;
          }
        }
        
        grad_w[l] = malloc(nodes[l-1] * nodes[l] * sizeof(double));
        for (i = 0; i < nodes[l-1] * nodes[l]; i++) {
          grad_w[l][i] = 0.0;
        }
        
      }
      break;
    default:
      break;
  }
  
  // delta_out dim(batch * columns_Y)
  deltas[layers] = malloc(batch * columns_Y * sizeof(double));
  deltas_row_sum[layers] = malloc(columns_Y * sizeof(double));
  
  for (i = 0; i < batch * columns_Y; i++) {
    deltas[layers][i] = 0.0;
    switch (i < columns_Y) {
      case 1:
        deltas_row_sum[layers][i] = 0.0;
        break;
      default:
        break;
    }
  }
  
  // grad_w allocation (same dimensions as wb[0][l])
  grad_w[layers] = malloc(nodes[layers-1] * columns_Y * sizeof(double));
  for (i = 0; i < nodes[layers-1] * columns_Y; i++) {
    grad_w[layers][i] = 0.0;
  }
  
  // Big switch in case we have pure batch (do not allocate mini-batches)
  switch (batch == rows) {
    // True
    case 1:
      correction = learning_rate * 1.0/(double)rows;
      
      // The training
      while (e != 0) {
        printf("%.10lf\n", loss);
        // Find deltas
        delta_gd(deltas, rows, columns_Y, layers, Y, Z, wb, nodes, funcs);
        
        // Now we update the weights and biases
        
        // Input layer
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    columns_X, // Rows of grad_w[l]
                    nodes[0], // Columns of grad_w[l]
                    rows, // columns of A, rows of B
                    1.0, // scaling factor
                    X, columns_X, // C = A * B -> matrix A, ldA
                    deltas[0], nodes[0], // C = A * B -> matrix B, ldB
                    1.0, // scaling factor for C (none)
                    grad_w[0], nodes[0]); // C, ldC
        
        row_sum(deltas_row_sum[0], deltas[0], rows, nodes[0]);
        
        while (for_helper_w >= 0) {
          wb[0][0][for_helper_w--] -= correction * grad_w[0][for_helper_w];
        }
        
        j = nodes[0];
        while (j >= 0) {
          wb[1][0][j--] -= correction * deltas_row_sum[0][j];
        }
        
        // Intermediate layers (if more than one hidden)
        switch (layers == 1) {
          // 1 hidden layer
          case 0:
            break;
          default:
            l = layers - 1;
            while (l >= 1) {
              for_helper_w = nodes[l-1] * nodes[l];
              
              cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                          nodes[l-1], // Rows of z[0][l][i][j]
                          nodes[l], // Columns of z[0][l][i][j]
                          rows, // columns of A, rows of B
                          1.0, // scaling factor (none)
                          Z[1][l-1], nodes[l-1], // C = A * B -> matrix A, ldA
                          deltas[l], nodes[l], // C = A * B -> matrix B, ldB
                          1.0, // scaling factor for C
                          grad_w[l], nodes[l]); // C, ldC
              
              row_sum(deltas_row_sum[l], deltas[l], rows, nodes[l]);
              
              while (for_helper_w >= 0) {
                wb[0][l][for_helper_w--] -= correction * grad_w[l][for_helper_w];
              }
              
              j = nodes[l];
              while (j >= 0) {
                wb[1][l][j--] -= correction * deltas_row_sum[l][j];
              }
              l--;
            }
            break;
        }
        
        // Output layer
        for_helper_w = nodes[layers-1] * columns_Y;
        
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    nodes[layers-1], // Rows of grad_w[l]
                    columns_Y, // Columns of grad_w[l]
                    rows, // columns of A, rows of B
                    1.0, // scaling factor
                    Z[1][layers-1], nodes[layers-1], // C = A * B -> matrix A, ldA
                    deltas[layers], columns_Y, // C = A * B -> matrix B, ldB
                    1.0, // scaling factor for C (none)
                    grad_w[layers], columns_Y); // C, ldC
        
        row_sum(deltas_row_sum[layers], deltas[layers], rows, columns_Y);
        
        while (for_helper_w >= 0) {
          wb[0][layers][for_helper_w--] -= correction * grad_w[layers][for_helper_w];
        }
        
        j = columns_Y;
        while (j >= 0) {
          wb[1][layers][j--] -= correction * deltas_row_sum[layers][j];
        }
        
        // Update Zs with the new wb's
        feedforward_update(Z, rows, columns_Y, columns_X, layers, X, wb, nodes, funcs);
        
        loss = rmse(rows, columns_Y, Y, Z[1][layers]);
        if (loss != loss) {
          printf("\nWe got NaN values at epoch = %d, aborting...\n", epochs - e);
          for (l = 0; l < layers + 1; l++) {
            free(deltas[l]);
            free(deltas_row_sum[l]);
            free(grad_w[l]);
          }
          free(deltas);
          free(deltas_row_sum);
          free(grad_w);
          return;
        }
        --e;
      }
      break;
    default:
      correction = learning_rate * 1.0/(double)batch;
      
      // They need to be allocated once
      double **X_batch = malloc(r_over_b * sizeof(double *));
      double **Y_batch = malloc(r_over_b * sizeof(double *));
      
      for (i = 0; i < r_over_b; i++) {
        X_batch[i] = malloc(batch * columns_X * sizeof(double));
        Y_batch[i] = malloc(batch * columns_Y * sizeof(double));
      }
      
      i = 0;
      for (b = 0; b < rows; b+=batch) {
        for (r = 0; r < batch; r++) {
          for (j = 0; j < columns_X; j++) {
            X_batch[i][r * columns_X + j] = X[(r + b) * columns_X + j];
          }
        }
        for (r = 0; r < batch; r++) {
          for (j = 0; j < columns_Y; j++) {
            Y_batch[i][r * columns_Y + j] = Y[(r + b) * columns_Y + j];
          }
        }
        i++;
      }
      
      // The perceptrons' outputs
      double ***Z_batch = malloc(2 * sizeof(double **));
      // Unactivated
      Z_batch[0] = malloc((layers + 1) * sizeof(double *));
      // Activated
      Z_batch[1] = malloc((layers + 1) * sizeof(double *));
      
      for (l = 0; l < layers + 1; l++) {
        switch (l == layers) {
          // Last layer
          case 1:
            Z_batch[0][l] = malloc(batch * columns_Y * sizeof(double));
            Z_batch[1][l] = malloc(batch * columns_Y * sizeof(double));
            break;
          default:
            Z_batch[0][l] = malloc(batch * nodes[l] * sizeof(double));
            Z_batch[1][l] = malloc(batch * nodes[l] * sizeof(double));
            break;
        }
      }
      
      while (e != 0) {
        printf("%.10lf\n", loss);
        
        i = 0;
        for (b = 0; b < rows; b+=batch) {
          // Fill X_batch, Y_batch, Z_batch
          for (r = 0; r < batch; r++) {
            for (j = 0; j < nodes[0]; j++) {
              Z_batch[0][0][r * nodes[0] + j] = Z[0][0][(r + b) * nodes[0] + j];
              Z_batch[1][0][r * nodes[0] + j] = Z[1][0][(r + b) * nodes[0] + j];
            }
            switch (layers == 1) {
              case 1:
                break;
              default:
                for (l = 1; l < layers; l++) {
                  for (j = 0; j < nodes[l]; j++) {
                    Z_batch[0][l][r * nodes[l] + j] = Z[0][l][(r + b) * nodes[l] + j];
                    Z_batch[1][l][r * nodes[l] + j] = Z[1][l][(r + b) * nodes[l] + j];
                  }
                }
                break;
            }
            for (j = 0; j < columns_Y; j++) {
              Z_batch[0][layers][r * columns_Y + j] = Z[0][layers][(r + b) * columns_Y + j];
              Z_batch[1][layers][r * columns_Y + j] = Z[1][layers][(r + b) * columns_Y + j];
            }
          }
          
          // Fill the deltas
          delta_gd(deltas, batch, columns_Y, layers, Y_batch[i], Z_batch, wb, nodes, funcs);
          
          // Now we update the weights and biases
          
          // Input layer
          for_helper_w = columns_X * nodes[0];
          
          cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                      columns_X, // Rows of grad_w[l]
                      nodes[0], // Columns of grad_w[l]
                      batch, // columns of A, rows of B
                      1.0, // scaling factor
                      X_batch[i], columns_X, // C = A * B -> matrix A, ldA
                      deltas[0], nodes[0], // C = A * B -> matrix B, ldB
                      1.0, // scaling factor for C (none)
                      grad_w[0], nodes[0]); // C, ldC
          
          row_sum(deltas_row_sum[0], deltas[0], batch, nodes[0]);
          
          while (for_helper_w >= 0) {
            wb[0][0][for_helper_w--] -= correction * grad_w[0][for_helper_w];
          }
          
          j = nodes[0];
          while (j >= 0) {
            wb[1][0][j--] -= correction * deltas_row_sum[0][j];
          }
          
          // Intermediate layers (if they exist i.e. more than one hidden layer)
          switch (layers > 1) {
            case 0:
              break;
            default:
              l = layers - 1;
              while (l >= 1) {
                for_helper_w = nodes[l-1] * nodes[l];
                
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            nodes[l-1], // Rows of z[0][l][i][j]
                            nodes[l], // Columns of z[0][l][i][j]
                            batch, // columns of A, rows of B
                            1.0, // scaling factor (none)
                            Z_batch[1][l-1], nodes[l-1], // matrix A, ldA
                            deltas[l], nodes[l], // C = A * B -> matrix B, ldB
                            1.0, // scaling factor for C
                            grad_w[l], nodes[l]); // C, ldC
                
                row_sum(deltas_row_sum[l], deltas[l], batch, nodes[l]);
                
                while (for_helper_w >= 0) {
                  wb[0][l][for_helper_w--] -= correction * grad_w[l][for_helper_w];
                }
                
                j = nodes[l];
                while (j >= 0) {
                  wb[1][l][j--] -= correction * deltas_row_sum[l][j];
                }
                l--;
              }
              break;
          }
          
          // Output layer
          for_helper_w = nodes[layers-1] * columns_Y;
          
          cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                      nodes[layers-1], // Rows of grad_w[l]
                      columns_Y, // Columns of grad_w[l]
                      batch, // columns of A, rows of B
                      1.0, // scaling factor
                      Z_batch[1][layers-1], nodes[layers-1],
                      deltas[layers], columns_Y, // C = A * B -> matrix B, ldB
                      1.0, // scaling factor for C (none)
                      grad_w[layers], columns_Y); // C, ldC
          
          row_sum(deltas_row_sum[layers], deltas[layers], batch, columns_Y);
          
          while (for_helper_w >= 0) {
            wb[0][layers][for_helper_w--] -= correction * grad_w[layers][for_helper_w];
          }
          
          j = columns_Y;
          while (j >= 0) {
            wb[1][layers][j--] -= correction * deltas_row_sum[layers][j];
          }
          
          // Do an update of Z's with the new wb's
          feedforward_update(Z, rows, columns_Y, columns_X, layers, X, wb, nodes, funcs);
          i++;
        }
        loss = rmse(rows, columns_Y, Y, Z[1][layers]);
        if (loss != loss) {
          printf("\nWe got NaN values at epoch = %d, aborting...\n", epochs - e);
          // Free before quitting
          for (i = 0; i < rows/batch; i++) {
            free(X_batch[i]);
            free(Y_batch[i]);
          }
          for (l = 0; l < layers + 1; l++) {
            free(Z_batch[0][l]);
            free(Z_batch[1][l]);
            free(deltas[l]);
            free(deltas_row_sum[l]);
            free(grad_w[l]);
          }
          free(Z_batch[0]);
          free(Z_batch[1]);
          free(Z_batch);
          free(Y_batch);
          free(X_batch);
          free(deltas);
          free(deltas_row_sum);
          free(grad_w);
          return;
        }
        --e;
      }
      
      // End of batching
      for (i = 0; i < rows/batch; i++) {
        free(X_batch[i]);
        free(Y_batch[i]);
      }
      for (l = 0; l < layers + 1; l++) {
        free(Z_batch[0][l]);
        free(Z_batch[1][l]);
      }
      free(Z_batch[0]);
      free(Z_batch[1]);
      free(Z_batch);
      free(Y_batch);
      free(X_batch);
      break;
  }
  
  // Free the rest
  for (l = 0; l < layers + 1; l++) {
    free(deltas[l]);
    free(deltas_row_sum[l]);
    free(grad_w[l]);
  }
  free(deltas);
  free(deltas_row_sum);
  free(grad_w);
}
