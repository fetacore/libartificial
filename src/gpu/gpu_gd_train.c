#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS // to disable deprecation warnings
#include "../../clblast/include/clblast_c.h"

#include "../headers/utils.h"
#include "../headers/gpu.h"

#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define RESET "\033[0m"

// GRADIENT DESCENT with batch size
// Directly manipulates weights and prediction

// For stochastic set batch = 1
// For mini-batch set batch = whatever you want (perfect divisor of rows)
// For pure batch gd set batch = rows

void gpu_threaded_batch(const double *Z_0, const double *Z_1, double *Z_batch_0, double *Z_batch_1,
                        const int *b, const int *cols, const int *batch)
{
  int r = (*batch) - 1, j = (*cols) - 1;
  do {
    Z_batch_0[r * (*cols) + j] = Z_0[(r + (*b)) * (*cols) + j];
    Z_batch_1[r * (*cols) + j] = Z_1[(r + (*b)) * (*cols) + j];
    switch (j == 0) {
      case 1:
        j = (*cols) - 1;
        break;
      default:
        --j;
        break;
    }
  } while (--r >= 0);
}

// returns new wb
void gpu_threaded_update(const double *X, const double *delta, double *grad_w, double *w,
                         const int *m, const int *n, const int *k, const double *correction,
                         cl_context *context, cl_device_id *device)
{
  int i = (*m) * (*n) - 1;
  
  cl_command_queue queue = clCreateCommandQueue((*context), (*device), 0, NULL);
  cl_event event = NULL;
  CLBlastStatusCode status;
  
  cl_mem dev_a = clCreateBuffer((*context), CL_MEM_READ_WRITE, (*k) * (*m) * sizeof(double), NULL, NULL);
  cl_mem dev_b = clCreateBuffer((*context), CL_MEM_READ_WRITE, (*k) * (*n) * sizeof(double), NULL, NULL);
  cl_mem dev_c = clCreateBuffer((*context), CL_MEM_READ_WRITE, (*m) * (*n) * sizeof(double), NULL, NULL);
  
  clEnqueueWriteBuffer(queue, dev_a, CL_TRUE, 0, (*k) * (*m) * sizeof(double), X, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, dev_b, CL_TRUE, 0, (*k) * (*n) * sizeof(double), delta, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, dev_c, CL_TRUE, 0, (*m) * (*n) * sizeof(double), grad_w, 0, NULL, NULL);
  
  status = CLBlastDgemm(CLBlastLayoutRowMajor, CLBlastTransposeYes, CLBlastTransposeNo,
                        (*m), (*n), (*k),
                        1.0,
                        dev_a, 0, (*m),
                        dev_b, 0, (*n),
                        0.0,
                        dev_c, 0, (*n),
                        &queue, &event);
  
  // Wait for completion
  if (status == CLBlastSuccess) {
    clWaitForEvents(1, &event);
    clEnqueueReadBuffer(queue, dev_c, CL_TRUE, 0, (*m) * (*n) * sizeof(double), grad_w, 0, NULL, NULL);
    clReleaseEvent(event);
    clReleaseMemObject(dev_a);
    clReleaseMemObject(dev_b);
    clReleaseMemObject(dev_c);
    clReleaseCommandQueue(queue);
  }
  
  do {
    w[i] -= grad_w[i];
  } while (--i >= 0);
}

void gpu_gd_train(const int rows, const int columns_Y, const int columns_X,
                  const int batch, const int layers, const int nodes[layers],
                  double *Y, double *X, double **w,
                  char functions[layers+1][30],
                  const double learning_rate, const int epochs)
{
  // l for layers
  // i for rows
  // j for columns
  // e for epochs
  int l, i, j, for_helper_w, for_helper_batch, e = epochs;
  int *funcs;
  funcs = name2int(layers, functions);
  
  omp_set_num_threads(layers + 1);
  
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
  
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
  
  // In case of mini-batching or stochastic
  // b for batch
  // r helper for batch rows
  int b, r, r_over_b = rows/batch;
  
  double loss = 0.0;
  
  // For the averaging of deltas in batch/mini-batch
  double correction = 1.0;
  
  double **deltas = malloc((layers + 1) * sizeof(double *));
  // The values to be subtracted from weights
  double **grad_w = malloc((layers + 1) * sizeof(double *));
  if (deltas && grad_w) {
    // Allocations
    for (l = 0; l < layers + 1; l++) {
      if (l == 0) {
        for_helper_batch = batch * nodes[l];
        for_helper_w = columns_X * nodes[l];
      } else if (l == layers) {
        for_helper_batch = batch * columns_Y;
        for_helper_w = nodes[l-1] * columns_Y;
      } else {
        for_helper_batch = batch * nodes[l];
        for_helper_w = nodes[l-1] * nodes[l];
      }
      deltas[l] = malloc(for_helper_batch * sizeof(double));    
      grad_w[l] = malloc(for_helper_w * sizeof(double));
      if (deltas[l] && grad_w[l]) {
        memset(deltas[l], 0.0, for_helper_batch * sizeof(double));
        memset(grad_w[l], 0.0, for_helper_w * sizeof(double));
      } else {
        printf(KRED "\nFailed to allocate deltas or gradients. Aborting...\n" RESET);
        free(deltas);
        free(grad_w);
        return;
      }
    }
  } else {
    printf(KRED "\nFailed to allocate deltas or gradients. Aborting...\n" RESET);
    return;
  }
  
  double ***Z;
  Z = gpu_feedforward_cache(&rows, &columns_Y, &columns_X, &layers, X, w, nodes, funcs, &context, &device);
  
  // Big switch in case we have pure batch (do not allocate mini-batches)
  switch (batch == rows) {
    // True
    case 1:
      correction = learning_rate * 1.0/(double)rows;
      
      // Training
      do {
        // Find deltas
        gpu_gd_delta(deltas, &rows, &columns_Y, &layers, Y, Z, w, nodes, funcs, &context, &device);
        
        // Now we update the weights and biases
        #pragma omp parallel for
        for (l = 0; l < layers + 1; l++) {
          if (l == 0) {
            gpu_threaded_update(X, deltas[l], grad_w[l], w[l], &columns_X, &nodes[l], &rows, &correction, &context, &device);
          } else if (l > 0 && l < layers) {
            gpu_threaded_update(Z[1][l-1], deltas[l], grad_w[l], w[l], &nodes[l-1], &nodes[l], &rows, &correction, &context, &device);
          } else {
            gpu_threaded_update(Z[1][l-1], deltas[l], grad_w[l], w[l], &nodes[l-1], &columns_Y, &rows, &correction, &context, &device);
          }
        }
        
        // Update Zs with the new wb's
        gpu_feedforward_update(&rows, &columns_Y, &columns_X, &layers, Z, X, w, nodes, funcs, &context, &device);
        
        loss = rmse(rows, columns_Y, Y, Z[1][layers]);
        
        if (loss != loss) {
          printf(KRED "\nWe got NaN values during training. Aborting...\n" RESET);
          for (l = 0; l < layers + 1; l++) {
            free(deltas[l]);
            free(grad_w[l]);
          }
          free(deltas);
          free(grad_w);
          free(funcs);
          delete_Z(layers, Z);
          clReleaseContext(context);
          free(platforms);
          free(devices);
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
        for (i = 0; i < r_over_b; i++) {
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
      
      i = 0;
      // Fill X_batch, Y_batch
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
      if (Z_batch) {
        // Unactivated
        Z_batch[0] = malloc((layers + 1) * sizeof(double *));
        // Activated
        Z_batch[1] = malloc((layers + 1) * sizeof(double *));
        if (Z_batch[0] && Z_batch[1]) {
          for (l = 0; l < layers + 1; l++) {
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
        i = 0;
        for (b = 0; b < rows; b+=batch) {
          if (b == 0) {
            printf("\nBatch\t#%d/%d\n", i + 1, r_over_b);
          } else {
            printf("     \t#%d/%d\n", i + 1, r_over_b);
          }
          
          // Fill Z_batch with new Zs
          #pragma omp parallel for
          for (l = 0; l < layers + 1; l++) {
            if (l == layers) {
              gpu_threaded_batch(Z[0][l], Z[1][l], Z_batch[0][l], Z_batch[1][l], &b, &columns_Y, &batch);
            } else {
              gpu_threaded_batch(Z[0][l], Z[1][l], Z_batch[0][l], Z_batch[1][l], &b, &nodes[l], &batch);
            }
          }
          
          // Fill the deltas
          gpu_gd_delta(deltas, &batch, &columns_Y, &layers, Y_batch[i], Z_batch, w, nodes, funcs, &context, &device);
          
          // Now we update the weights and biases
          #pragma omp parallel for
          for (l = 0; l < layers + 1; l++) {
            if (l == 0) {
              gpu_threaded_update(X, deltas[l], grad_w[l], w[l],
                                  &columns_X, &nodes[l], &batch, &correction, &context, &device);
            } else if (l > 0 && l < layers) {
              gpu_threaded_update(Z[1][l-1], deltas[l], grad_w[l], w[l],
                                  &nodes[l-1], &nodes[l], &batch, &correction, &context, &device);
            } else {
              gpu_threaded_update(Z[1][l-1], deltas[l], grad_w[l], w[l],
                                  &nodes[l-1], &columns_Y, &batch, &correction, &context, &device);
            }
          }
          
          // Do an update of Z's with the new wb's
          gpu_feedforward_update(&rows, &columns_Y, &columns_X, &layers, Z, X, w, nodes, funcs, &context, &device);
          
          ++i;
        }
        
        loss = rmse(rows, columns_Y, Y, Z[1][layers]);
        
        if (loss != loss) {
          printf(KRED "\nWe got NaN values during training. Aborting...\n" RESET);
          // Free before quitting
          for (l = 0; l < layers + 1; l++) {
            free(Z_batch[0][l]);
            free(Z_batch[1][l]);
            free(deltas[l]);
            free(grad_w[l]);
          }
          free(Z_batch[0]);
          free(Z_batch[1]);
          free(Z_batch);
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
          clReleaseContext(context);
          free(platforms);
          free(devices);
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
  }
  free(deltas);
  free(grad_w);
  free(funcs);
  delete_Z(layers, Z);
  clReleaseContext(context);
  free(platforms);
  free(devices);
}

