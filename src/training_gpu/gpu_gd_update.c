#include <stdio.h>
#include <stdlib.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS // to disable deprecation warnings
#include "../../clblast/include/clblast_c.h"

#include "../headers/utils.h"
#include "../headers/training_gpu.h"

#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define RESET "\033[0m"

// GRADIENT DESCENT with batch size
// Directly manipulates weights and prediction

// For stochastic set batch = 1
// For mini-batch set batch = whatever you want (perfect divisor of rows)
// For pure batch gd set batch = rows

// Threaded creation of batches
struct gpu_batch_struct {
  double *Z_0;
  double *Z_1;
  double *Z_batch_0;
  double *Z_batch_1;
  size_t b;
  size_t cols;
  size_t batch;
};

static void *gpu_threaded_batch(void *arguments)
{
  struct gpu_batch_struct *args = (struct gpu_batch_struct *)arguments;
  size_t r, j;
  for (r = 0; r < args -> batch; r++) {
    for (j = 0; j < args -> cols; j++) {
      args -> Z_batch_0[r * args -> cols + j] = args -> Z_0[(r + args -> b) * args -> cols + j];
      args -> Z_batch_1[r * args -> cols + j] = args -> Z_1[(r + args -> b) * args -> cols + j];
    }
  }
}

// pthread requires struct for multiple arguments
struct gpu_update_struct {
  double *X;
  double *delta;
  double *grad_w;
  double *wb_0;
  double *wb_1;
  double *delta_sum;
  size_t m;
  size_t n;
  size_t k;
  double correction;
  cl_device_id device;
};

// returns new wb
static void *gpu_threaded_update(void *arguments)
{
  struct gpu_update_struct *args = (struct gpu_update_struct *)arguments;
  
  int i = args -> m * args -> n - 1, j = args -> n - 1;
  
  // Creates the OpenCL context, queue, and an event
  cl_context context = clCreateContext(NULL, 1, &args -> device, NULL, NULL, NULL);
  cl_command_queue queue = clCreateCommandQueue(context, args -> device, 0, NULL);
  cl_event event = NULL;
  
  cl_mem device_a = clCreateBuffer(context, CL_MEM_READ_WRITE, args -> k * args -> m * sizeof(double), NULL, NULL);
  cl_mem device_b = clCreateBuffer(context, CL_MEM_READ_WRITE, args -> k * args -> n * sizeof(double), NULL, NULL);
  cl_mem device_c = clCreateBuffer(context, CL_MEM_READ_WRITE, args -> m * args -> n * sizeof(double), NULL, NULL);
  
  clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, args -> k * args -> m * sizeof(double),
                       args -> X, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, args -> k * args -> n * sizeof(double),
                       args -> delta, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, device_c, CL_TRUE, 0, args -> m * args -> n * sizeof(double),
                       args -> grad_w, 0, NULL, NULL);
  
  CLBlastStatusCode status = CLBlastDgemm(CLBlastLayoutRowMajor, CLBlastTransposeYes, CLBlastTransposeNo,
                                          args -> m, args -> n, args -> k,
                                          1.0,
                                          device_a, 0, args -> m,
                                          device_b, 0, args -> n,
                                          0.0,
                                          device_c, 0, args -> n,
                                          &queue, &event);
  
  // Wait for completion
  if (status == CLBlastSuccess) {
    clWaitForEvents(1, &event);
    clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, args -> m * args -> n * sizeof(double),
                        args -> grad_w, 0, NULL, NULL);
    clReleaseEvent(event);
    clReleaseMemObject(device_a);
    clReleaseMemObject(device_b);
    clReleaseMemObject(device_c);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
  }
  
  row_sum(args -> delta_sum, args -> delta, args -> k, args -> n);
  
  do {
    args -> wb_0[i] -= args -> grad_w[i];
    i--;
  } while (i >= 0);
  
  do {
    args -> wb_1[j] -= args -> correction * args -> delta_sum[j];
    j--;
  } while (j >= 0);
}

void gpu_gd_update(const size_t rows, const size_t columns_Y, const size_t columns_X,
                   const int batch, const int layers, const int nodes[layers],
                   double *Y, double *X, double ***Z, double ***wb,
                   char funcs[layers + 1][30], const double learning_rate, const int epochs)
{
  // l for layers
  // i for rows
  // j for columns
  // e for epochs
  int l, i, j, for_helper_w, for_helper_batch, stopper, e = epochs;
  
  // To pass values to thread
  struct gpu_update_struct args;
  struct gpu_batch_struct bargs;
  
  // In case of mini-batching or stochastic
  // b for batch
  // r helper for batch rows
  int b, r, r_over_b = (int)rows/batch;
  
  double loss = 0.0;
  
  // For the averaging of deltas in batch/mini-batch
  double correction = 1.0;
  
  // The values to be subtracted from weights
  double **grad_w = calloc(layers + 1, sizeof(double *));
  
  double **deltas = calloc(layers + 1, sizeof(double *));
  
  // For batch/mini-batch
  double **deltas_row_sum = calloc(layers + 1, sizeof(double *));
  
  // Allocations
  for (l = 0; l < layers + 1; l++) {
    if (l == 0) {
      for_helper_batch = batch * nodes[l];
      for_helper_w = (int)columns_X * nodes[l];
      stopper = nodes[l];
    } else if (l == layers) {
      for_helper_batch = batch * (int)columns_Y;
      for_helper_w = nodes[l-1] * (int)columns_Y;
      stopper = (int)columns_Y;
    } else {
      for_helper_batch = batch * nodes[l];
      for_helper_w = nodes[l-1] * nodes[l];
      stopper = nodes[l];
    }
    
    deltas[l] = calloc(for_helper_batch, sizeof(double));
    deltas_row_sum[l] = calloc(stopper, sizeof(double));
    
    grad_w[l] = calloc(for_helper_w, sizeof(double));
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
  
  // Big switch in case we have pure batch (do not allocate mini-batches)
  switch (batch == (int)rows) {
    // True
    case 1:
      printf(KGRN "Succesfully allocated space! Updating starts...\n" RESET);
      correction = learning_rate * 1.0/(double)rows;
      
      // The training
      do {
        // Find deltas
        gpu_gd_delta(deltas, rows, columns_Y, layers, Y, Z, wb, nodes, funcs, device);
        
        // Now we update the weights and biases
        args.k = rows;
        args.correction = correction;
        args.device = device;
        for (l = 0; l < layers + 1; l++) {
          args.delta = deltas[l];
          args.grad_w = grad_w[l];
          args.delta_sum = deltas_row_sum[l];
          args.wb_0 = wb[0][l];
          args.wb_1 = wb[1][l];
          if (l == 0) {
            args.m = columns_X;
            args.n = nodes[l];
            args.X = X;
          } else if (l > 0 && l < layers) {
            args.m = nodes[l-1];
            args.n = nodes[l];
            args.X = Z[1][l-1];
          } else {
            args.m = nodes[l-1];
            args.n = columns_Y;
            args.X = Z[1][l-1];
          }
          gpu_threaded_update((void *)&args);
        }
        
        // Update Zs with the new wb's
        gpu_feedforward_update(rows, columns_Y, columns_X, layers, Z, X, wb, nodes, funcs, device);
        
        loss = rmse(rows, columns_Y, Y, Z[1][layers]);
        if (loss != loss) {
          printf(KRED "\nWe got NaN values at epoch = %d, aborting...\n" RESET, epochs - e);
          for (l = 0; l < layers + 1; l++) {
            free(deltas[l]);
            free(deltas_row_sum[l]);
            free(grad_w[l]);
          }
          free(deltas);
          free(deltas_row_sum);
          free(grad_w);
          free(platforms);
          free(devices);
          return;
        }
        printf("\nLoss = %.10lf at epoch = %d\n", loss, epochs - e);
        e--;
      } while (e != 0);
      break;
      default:
        correction = learning_rate * 1.0/(double)batch;
        
        // They need to be allocated once
        double **X_batch = malloc(r_over_b * sizeof(double *));
        double **Y_batch = malloc(r_over_b * sizeof(double *));
        
        for (i = 0; i < r_over_b; i++) {
          X_batch[i] = malloc(batch * (int)columns_X * sizeof(double));
          Y_batch[i] = malloc(batch * (int)columns_Y * sizeof(double));
        }
        
        i = 0;
        // Fill X_batch, Y_batch
        for (b = 0; b < (int)rows; b+=batch) {
          for (r = 0; r < batch; r++) {
            for (j = 0; j < (int)columns_X; j++) {
              X_batch[i][r * (int)columns_X + j] = X[(r + b) * (int)columns_X + j];
            }
          }
          for (r = 0; r < batch; r++) {
            for (j = 0; j < (int)columns_Y; j++) {
              Y_batch[i][r * (int)columns_Y + j] = Y[(r + b) * (int)columns_Y + j];
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
              Z_batch[0][l] = malloc(batch * (int)columns_Y * sizeof(double));
              Z_batch[1][l] = malloc(batch * (int)columns_Y * sizeof(double));
              continue;
            default:
              Z_batch[0][l] = malloc(batch * nodes[l] * sizeof(double));
              Z_batch[1][l] = malloc(batch * nodes[l] * sizeof(double));
              continue;
          }
        }
        
        printf(KGRN "Succesfully allocated space! Updating starts...\n" RESET);
        
        do {
          i = 0;
          for (b = 0; b < (int)rows; b+=batch) {
            if (b == 0) {
              printf("\nBatch\t#%d/%d\n", i + 1, r_over_b);
            } else {
              printf("     \t#%d/%d\n", i + 1, r_over_b);
            }
            
            // Fill Z_batch with new Zs
            for (l = 0; l < layers + 1; l++) {
              bargs.b = b;
              bargs.batch = batch;
              bargs.Z_0 = Z[0][l];
              bargs.Z_1 = Z[1][l];
              bargs.Z_batch_0 = Z_batch[0][l];
              bargs.Z_batch_1 = Z_batch[1][l];
              if (l == layers) {
                bargs.cols = columns_Y;
              } else {
                bargs.cols = nodes[l];
              }
              gpu_threaded_batch((void *)&bargs);
            }
            
            // Fill the deltas
            gpu_gd_delta(deltas, batch, columns_Y, layers, Y_batch[i], Z_batch, wb, nodes, funcs, device);
            
            // Now we update the weights and biases
            args.k = batch;
            args.correction = correction;
            args.device = device;
            for (l = 0; l < layers + 1; l++) {
              args.delta = deltas[l];
              args.grad_w = grad_w[l];
              args.delta_sum = deltas_row_sum[l];
              args.wb_0 = wb[0][l];
              args.wb_1 = wb[1][l];
              if (l == 0) {
                args.m = columns_X;
                args.n = nodes[l];
                args.X = X_batch[i];
              } else if (l > 0 && l < layers) {
                args.m = nodes[l-1];
                args.n = nodes[l];
                args.X = Z_batch[1][l-1];
              } else {
                args.m = nodes[l-1];
                args.n = columns_Y;
                args.X = Z_batch[1][l-1];
              }
              gpu_threaded_update((void *)&args);
            }
            
            // Do an update of Z's with the new wb's
            gpu_feedforward_update(rows, columns_Y, columns_X, layers, Z, X, wb, nodes, funcs, device);
            
            i++;
          }
          
          loss = rmse(rows, columns_Y, Y, Z[1][layers]);
          
          if (loss != loss) {
            printf(KRED "\nWe got NaN values at epoch = %d, aborting...\n" RESET, epochs - e);
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
            free(platforms);
            free(devices);
            return;
          }
          printf("\nLoss = %.10lf at epoch = %d\n", loss, epochs - e);
          e--;
        } while (e != 0);
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
  
  // Free the rest
  for (l = 0; l < layers + 1; l++) {
    free(deltas[l]);
    free(deltas_row_sum[l]);
    free(grad_w[l]);
  }
  free(deltas);
  free(deltas_row_sum);
  free(grad_w);
  free(platforms);
  free(devices);
}
