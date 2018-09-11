#include <stdio.h>
#include <stdlib.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS // to disable deprecation warnings
#include "../../clblast/include/clblast_c.h"

#include "../headers/utils.h"
#include "../headers/training_gpu.h"

// GRADIENT DESCENT with batch size
// Directly manipulates weights and prediction

// For stochastic set batch = 1
// For mini-batch set batch = whatever you want (perfect divisor of rows)
// For pure batch gd set batch = rows

void update_gd_gpu(const size_t rows, const size_t columns_Y, const size_t columns_X,
                   const int batch, const int layers, const int nodes[layers],
                   const double *Y, const double *X, double ***Z, double ***wb,
                   char funcs[layers + 1][30],
                   const double learning_rate, const int epochs)
{
  // l for layers
  // i for rows
  // j for columns
  // e for epochs
  int l, i, j, e = epochs;
  
  // In case of mini-batching or stochastic
  // b for batch
  // r helper for batch rows
  int b, r, r_over_b = rows/batch, for_helper_batch = batch * nodes[0], for_helper_w = columns_X * nodes[0];
  
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
  deltas[0] = malloc(for_helper_batch * sizeof(double));
  deltas_row_sum[0] = malloc(nodes[0] * sizeof(double));
  
  for (i = 0; i < for_helper_batch; i++) {
    deltas[0][i] = 0.0;
    switch (i < nodes[0]) {
      case 1:
        deltas_row_sum[0][i] = 0.0;
        continue;
      default:
        continue;
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
        for_helper_w = nodes[l-1] * nodes[l];
        // delta dim (batch * nodes[l])
        for_helper_batch = batch * nodes[l];
        deltas[l] = malloc(for_helper_batch * sizeof(double));
        deltas_row_sum[l] = malloc(nodes[l] * sizeof(double));
        
        for (i = 0; i < for_helper_batch; i++) {
          deltas[l][i] = 0.0;
          switch (i < nodes[l]) {
            case 1:
              deltas_row_sum[l][i] = 0.0;
              continue;
            default:
              continue;
          }
        }
        
        grad_w[l] = malloc(for_helper_w * sizeof(double));
        for (i = 0; i < for_helper_w; i++) {
          grad_w[l][i] = 0.0;
        }
        
      }
      break;
    default:
      break;
  }
  
  for_helper_w = nodes[layers-1] * columns_Y;
  // delta_out dim(batch * columns_Y)
  for_helper_batch = batch * columns_Y;
  deltas[layers] = malloc(for_helper_batch * sizeof(double));
  deltas_row_sum[layers] = malloc(columns_Y * sizeof(double));
  
  for (i = 0; i < batch * (int)columns_Y; i++) {
    deltas[layers][i] = 0.0;
    switch (i < (int)columns_Y) {
      case 1:
        deltas_row_sum[layers][i] = 0.0;
        continue;
      default:
        continue;
    }
  }
  
  // grad_w allocation (same dimensions as wb[0][l])
  grad_w[layers] = malloc(for_helper_w * sizeof(double));
  for (i = 0; i < for_helper_w; i++) {
    grad_w[layers][i] = 0.0;
  }
  
  // OpenCL platform/device settings
  const size_t platform_id = 0;
  const size_t device_id = 0;
  
  // Initializes the OpenCL platform
  cl_uint num_platforms;
  clGetPlatformIDs(0, NULL, &num_platforms);
  cl_platform_id *platforms = malloc(num_platforms * sizeof(cl_platform_id));
  clGetPlatformIDs(num_platforms, platforms, NULL);
  cl_platform_id platform = platforms[platform_id];
  
  // Initializes the OpenCL device
  cl_uint num_devices;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
  cl_device_id *devices = malloc(num_devices * sizeof(cl_device_id));
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
  cl_device_id device = devices[device_id];
  
  // Creates the OpenCL context, queue, and an event
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
  cl_event event = NULL;
  
  cl_mem device_a, device_b, device_c;
  
  // Big switch in case we have pure batch (do not allocate mini-batches)
  switch (batch == (int)rows) {
    // True
    case 1:
      correction = learning_rate * 1.0/(double)rows;
      
      // The training
      while (e != 0) {
        if (loss != 0.0) {
          printf("%.10lf\n", loss);
        }
        // Find deltas
        delta_gd_gpu(deltas, rows, columns_Y, layers, Y, Z, wb, nodes, funcs, device);
        
        // Now we update the weights and biases
        
        // Copy the matrices to the device
        device_a = clCreateBuffer(context, CL_MEM_READ_WRITE, rows * columns_X * sizeof(double), NULL, NULL);
        device_b = clCreateBuffer(context, CL_MEM_READ_WRITE, rows * nodes[0] * sizeof(double), NULL, NULL);
        device_c = clCreateBuffer(context, CL_MEM_READ_WRITE, columns_X * nodes[0] * sizeof(double), NULL, NULL);
        clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, rows * columns_X * sizeof(double),
                             X, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, rows * nodes[0] * sizeof(double),
                             deltas[0], 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, device_c, CL_TRUE, 0, columns_X * nodes[0] * sizeof(double),
                             grad_w[0], 0, NULL, NULL);
        
        // Input layer
        CLBlastStatusCode status = CLBlastDgemm(CLBlastLayoutRowMajor, CLBlastTransposeYes, CLBlastTransposeNo,
                                                columns_X, nodes[0], rows,
                                                1.0,
                                                device_a, 0, columns_X,
                                                device_b, 0, nodes[0],
                                                0.0,
                                                device_c, 0, nodes[0],
                                                &queue, &event);
        
        // Wait for completion
        if (status == CLBlastSuccess) {
          clWaitForEvents(1, &event);
          clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, columns_X * nodes[0] * sizeof(double),
                              grad_w[0], 0, NULL, NULL);
          clReleaseEvent(event);
        }
        
        row_sum(deltas_row_sum[0], deltas[0], rows, nodes[0]);
        
        for_helper_w = columns_X * nodes[0];
        i = for_helper_w - 1;
        
        do {
          wb[0][0][i] -= correction * grad_w[0][i];
          i--;
        } while (i >= 0);
        
        j = nodes[0] - 1;
        do {
          wb[1][0][j] -= correction * deltas_row_sum[0][j];
          j--;
        } while (j >= 0);
        
        // Intermediate layers (if more than one hidden)
        switch (layers == 1) {
          // 1 hidden layer
          case 0:
            break;
          default:
            l = layers - 1;
            while (l >= 1) {
              for_helper_w = nodes[l-1] * nodes[l];
              
              // Copy the matrices to the device
              device_a = clCreateBuffer(context, CL_MEM_READ_WRITE, rows * nodes[l-1] * sizeof(double), NULL, NULL);
              device_b = clCreateBuffer(context, CL_MEM_READ_WRITE, rows * nodes[l] * sizeof(double), NULL, NULL);
              device_c = clCreateBuffer(context, CL_MEM_READ_WRITE, nodes[l-1] * nodes[l] * sizeof(double), NULL, NULL);
              clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, rows * nodes[l-1] * sizeof(double),
                                   Z[1][l-1], 0, NULL, NULL);
              clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, rows * nodes[l] * sizeof(double),
                                   deltas[l], 0, NULL, NULL);
              clEnqueueWriteBuffer(queue, device_c, CL_TRUE, 0, nodes[l-1] * nodes[l] * sizeof(double),
                                   grad_w[l], 0, NULL, NULL);
              
              status = CLBlastDgemm(CLBlastLayoutRowMajor, CLBlastTransposeYes, CLBlastTransposeNo,
                                    nodes[l-1], nodes[l], rows,
                                    1.0,
                                    device_a, 0, nodes[l-1],
                                    device_b, 0, nodes[l],
                                    0.0,
                                    device_c, 0, nodes[l],
                                    &queue, &event);
              
              // Wait for completion
              if (status == CLBlastSuccess) {
                clWaitForEvents(1, &event);
                clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, nodes[l-1] * nodes[l] * sizeof(double),
                                    grad_w[l], 0, NULL, NULL);
                clReleaseEvent(event);
              }
              
              row_sum(deltas_row_sum[l], deltas[l], rows, nodes[l]);
              
              i = for_helper_w - 1;
              do {
                wb[0][l][i] -= correction * grad_w[l][i];
                i--;
              } while (i >= 0);
              
              j = nodes[l] - 1;
              do {
                wb[1][l][j] -= correction * deltas_row_sum[l][j];
                j--;
              } while (j >= 0);
              l--;
            }
            break;
        }
                
        // Output layer
        for_helper_w = nodes[layers-1] * columns_Y;
        
        // Copy the matrices to the device
        device_a = clCreateBuffer(context, CL_MEM_READ_WRITE, rows * nodes[layers-1] * sizeof(double), NULL, NULL);
        device_b = clCreateBuffer(context, CL_MEM_READ_WRITE, rows * columns_Y * sizeof(double), NULL, NULL);
        device_c = clCreateBuffer(context, CL_MEM_READ_WRITE, nodes[layers-1] * columns_Y * sizeof(double), NULL, NULL);
        clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, rows * nodes[layers-1] * sizeof(double),
                             Z[1][layers-1], 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, rows * columns_Y * sizeof(double),
                             deltas[layers], 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, device_c, CL_TRUE, 0, nodes[layers-1] * columns_Y * sizeof(double),
                             grad_w[layers], 0, NULL, NULL);
        
        status = CLBlastDgemm(CLBlastLayoutRowMajor, CLBlastTransposeYes, CLBlastTransposeNo,
                              nodes[layers-1], columns_Y, rows,
                              1.0,
                              device_a, 0, nodes[layers-1],
                              device_b, 0, columns_Y,
                              0.0,
                              device_c, 0, columns_Y,
                              &queue, &event);
        
        // Wait for completion
        if (status == CLBlastSuccess) {
          clWaitForEvents(1, &event);
          clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, nodes[layers-1] * columns_Y * sizeof(double),
                              grad_w[layers], 0, NULL, NULL);
          clReleaseEvent(event);
        }
        
        row_sum(deltas_row_sum[layers], deltas[layers], rows, columns_Y);
        
        i = for_helper_w - 1;
        do {
          wb[0][layers][i] -= correction * grad_w[layers][i];
          i--;
        } while (i >= 0);
        
        j = columns_Y - 1;
        do {
          wb[1][layers][j] -= correction * deltas_row_sum[layers][j];
          j--;
        } while (j >= 0);
        
        // Update Zs with the new wb's
        feedforward_update_gpu(Z, rows, columns_Y, columns_X, layers, X, wb, nodes, funcs, device);
        
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
          clReleaseMemObject(device_a);
          clReleaseMemObject(device_b);
          clReleaseMemObject(device_c);
          clReleaseCommandQueue(queue);
          clReleaseContext(context);
          free(platforms);
          free(devices);
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
      for (b = 0; b < (int)rows; b+=batch) {
        for (r = 0; r < batch; r++) {
          for (j = 0; j < (int)columns_X; j++) {
            X_batch[i][r * columns_X + j] = X[(r + b) * columns_X + j];
          }
        }
        for (r = 0; r < (int)batch; r++) {
          for (j = 0; j < (int)columns_Y; j++) {
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
            continue;
          default:
            Z_batch[0][l] = malloc(batch * nodes[l] * sizeof(double));
            Z_batch[1][l] = malloc(batch * nodes[l] * sizeof(double));
            continue;
        }
      }
      
      while (e != 0) {
        if (loss != 0.0) {
          printf("%.10lf\n", loss);
        }
        
        i = 0;
        for (b = 0; b < (int)rows; b+=batch) {
          // Fill X_batch, Y_batch, Z_batch
          for (r = 0; r < batch; r++) {
            for (j = 0; j < nodes[0]; j++) {
              Z_batch[0][0][r * nodes[0] + j] = Z[0][0][(r + b) * nodes[0] + j];
              Z_batch[1][0][r * nodes[0] + j] = Z[1][0][(r + b) * nodes[0] + j];
            }
            switch (layers == 1) {
              case 1:
                continue;
              default:
                for (l = 1; l < layers; l++) {
                  for (j = 0; j < nodes[l]; j++) {
                    Z_batch[0][l][r * nodes[l] + j] = Z[0][l][(r + b) * nodes[l] + j];
                    Z_batch[1][l][r * nodes[l] + j] = Z[1][l][(r + b) * nodes[l] + j];
                  }
                }
                continue;
            }
            for (j = 0; j < (int)columns_Y; j++) {
              Z_batch[0][layers][r * columns_Y + j] = Z[0][layers][(r + b) * columns_Y + j];
              Z_batch[1][layers][r * columns_Y + j] = Z[1][layers][(r + b) * columns_Y + j];
            }
          }
          
          // Fill the deltas
          delta_gd_gpu(deltas, batch, columns_Y, layers, Y_batch[i], Z_batch, wb, nodes, funcs, device);
          
          // Now we update the weights and biases
          // Copy the matrices to the device
          device_a = clCreateBuffer(context, CL_MEM_READ_WRITE, batch * columns_X * sizeof(double), NULL, NULL);
          device_b = clCreateBuffer(context, CL_MEM_READ_WRITE, batch * nodes[0] * sizeof(double), NULL, NULL);
          device_c = clCreateBuffer(context, CL_MEM_READ_WRITE, columns_X * nodes[0] * sizeof(double), NULL, NULL);
          clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, batch * columns_X * sizeof(double),
                               X_batch[i], 0, NULL, NULL);
          clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, batch * nodes[0] * sizeof(double),
                               deltas[0], 0, NULL, NULL);
          clEnqueueWriteBuffer(queue, device_c, CL_TRUE, 0, columns_X * nodes[0] * sizeof(double),
                               grad_w[0], 0, NULL, NULL);
          
          // Input layer
          CLBlastStatusCode status = CLBlastDgemm(CLBlastLayoutRowMajor, CLBlastTransposeYes, CLBlastTransposeNo,
                                                  columns_X, nodes[0], batch,
                                                  1.0,
                                                  device_a, 0, columns_X,
                                                  device_b, 0, nodes[0],
                                                  0.0,
                                                  device_c, 0, nodes[0],
                                                  &queue, &event);
          
          // Wait for completion
          if (status == CLBlastSuccess) {
            clWaitForEvents(1, &event);
            clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, columns_X * nodes[0] * sizeof(double),
                                grad_w[0], 0, NULL, NULL);
            clReleaseEvent(event);
          }
          
          // Input layer
          for_helper_w = columns_X * nodes[0];
          
          row_sum(deltas_row_sum[0], deltas[0], batch, nodes[0]);
          
          i = for_helper_w - 1;
          do {
            wb[0][0][i] -= correction * grad_w[0][i];
            i--;
          } while (i >= 0);
          
          j = nodes[0] - 1;
          do {
            wb[1][0][j] -= correction * deltas_row_sum[0][j];
            j--;
          } while (j >= 0);
          
          // Intermediate layers (if they exist i.e. more than one hidden layer)
          switch (layers > 1) {
            case 0:
              break;
            default:
              l = layers - 1;
              while (l >= 1) {
                for_helper_w = nodes[l-1] * nodes[l];
                
                // Copy the matrices to the device
                device_a = clCreateBuffer(context, CL_MEM_READ_WRITE, batch * nodes[l-1] * sizeof(double), NULL, NULL);
                device_b = clCreateBuffer(context, CL_MEM_READ_WRITE, batch * nodes[l] * sizeof(double), NULL, NULL);
                device_c = clCreateBuffer(context, CL_MEM_READ_WRITE, nodes[l-1] * nodes[l] * sizeof(double), NULL, NULL);
                clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, batch * nodes[l-1] * sizeof(double),
                                     Z_batch[1][l-1], 0, NULL, NULL);
                clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, batch * nodes[l] * sizeof(double),
                                     deltas[l], 0, NULL, NULL);
                clEnqueueWriteBuffer(queue, device_c, CL_TRUE, 0, nodes[l-1] * nodes[l] * sizeof(double),
                                     grad_w[l], 0, NULL, NULL);
                
                status = CLBlastDgemm(CLBlastLayoutRowMajor, CLBlastTransposeYes, CLBlastTransposeNo,
                                      nodes[l-1], nodes[l], batch,
                                      1.0,
                                      device_a, 0, nodes[l-1],
                                      device_b, 0, nodes[l],
                                      0.0,
                                      device_c, 0, nodes[l],
                                      &queue, &event);
                
                // Wait for completion
                if (status == CLBlastSuccess) {
                  clWaitForEvents(1, &event);
                  clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, nodes[l-1] * nodes[l] * sizeof(double),
                                      grad_w[l], 0, NULL, NULL);
                  clReleaseEvent(event);
                }
                
                row_sum(deltas_row_sum[l], deltas[l], batch, nodes[l]);
                
                i = for_helper_w - 1;
                do {
                  wb[0][l][i] -= correction * grad_w[l][i];
                  i--;
                } while (i >= 0);
                
                j = nodes[l] - 1;
                do {
                  wb[1][l][j] -= correction * deltas_row_sum[l][j];
                  j--;
                } while (j >= 0);
                l--;
              }
              break;
          }
          
          // Output layer
          for_helper_w = nodes[layers-1] * columns_Y;
          
          // Copy the matrices to the device
          device_a = clCreateBuffer(context, CL_MEM_READ_WRITE, batch * nodes[layers-1] * sizeof(double), NULL, NULL);
          device_b = clCreateBuffer(context, CL_MEM_READ_WRITE, batch * columns_Y * sizeof(double), NULL, NULL);
          device_c = clCreateBuffer(context, CL_MEM_READ_WRITE, nodes[layers-1] * columns_Y * sizeof(double), NULL, NULL);
          clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, batch * nodes[layers-1] * sizeof(double),
                               Z_batch[1][layers-1], 0, NULL, NULL);
          clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, batch * columns_Y * sizeof(double),
                               deltas[layers], 0, NULL, NULL);
          clEnqueueWriteBuffer(queue, device_c, CL_TRUE, 0, nodes[layers-1] * columns_Y * sizeof(double),
                               grad_w[layers], 0, NULL, NULL);
          
          status = CLBlastDgemm(CLBlastLayoutRowMajor, CLBlastTransposeYes, CLBlastTransposeNo,
                                nodes[layers-1], columns_Y, batch,
                                1.0,
                                device_a, 0, nodes[layers-1],
                                device_b, 0, columns_Y,
                                0.0,
                                device_c, 0, columns_Y,
                                &queue, &event);
          
          // Wait for completion
          if (status == CLBlastSuccess) {
            clWaitForEvents(1, &event);
            clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, nodes[layers-1] * columns_Y * sizeof(double),
                                grad_w[layers], 0, NULL, NULL);
            clReleaseEvent(event);
          }
          
          row_sum(deltas_row_sum[layers], deltas[layers], batch, columns_Y);
          
          i = for_helper_w - 1;
          do {
            wb[0][layers][i] -= correction * grad_w[layers][i];
            i--;
          } while (i >= 0);
          
          j = columns_Y - 1;
          do {
            wb[1][layers][j] -= correction * deltas_row_sum[layers][j];
            j--;
          } while (j >= 0);
          
          // Do an update of Z's with the new wb's
          feedforward_update_gpu(Z, rows, columns_Y, columns_X, layers, X, wb, nodes, funcs, device);
          i++;
        }
        
        loss = rmse(rows, columns_Y, Y, Z[1][layers]);

        if (loss != loss) {
          printf("\nWe got NaN values at epoch = %d, aborting...\n", epochs - e);
          // Free before quitting
          for (i = 0; i < (int)rows/batch; i++) {
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
          clReleaseMemObject(device_a);
          clReleaseMemObject(device_b);
          clReleaseMemObject(device_c);
          clReleaseCommandQueue(queue);
          clReleaseContext(context);
          free(platforms);
          free(devices);
          return;
        }
        e--;
      }
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
      clReleaseMemObject(device_a);
      clReleaseMemObject(device_b);
      clReleaseMemObject(device_c);
      clReleaseCommandQueue(queue);
      clReleaseContext(context);
      free(platforms);
      free(devices);
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
  clReleaseMemObject(device_a);
  clReleaseMemObject(device_b);
  clReleaseMemObject(device_c);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  free(platforms);
  free(devices);
}
