#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../headers/utils.h"
#include "../headers/gpu.h"

#define KRED  "\x1B[31m"
#define RESET "\033[0m"

double *gpu_feedforward_predict(const int rows, const int columns_Y, const int columns_X, const int layers,
                                double *X, double **w,
                                const int nodes[layers], char funcs[layers+1][30])
{
  // l is for layers
  // i is for each row * column of X, Y
  int l = layers;
  int r_times_col = rows * columns_Y;
  int *functions;
  
  double *Z_pred = malloc(r_times_col * sizeof(double));
  if (Z_pred) {
    memset(Z_pred, 0.0, r_times_col * sizeof(double));
  } else {
    printf(KRED "\nCould not allocate Z_prediction. Aborting...\n" RESET);
    abort();
  }
  
  // feeds at every layer
  double ***Z = malloc(2 * sizeof(double **));
  if (Z) {
    Z[0] = malloc((layers + 1) * sizeof(double *));
    Z[1] = malloc((layers + 1) * sizeof(double *));
    if (Z[0] && Z[1]) {
      Z[0][l] = malloc(r_times_col * sizeof(double));
      Z[1][l] = malloc(r_times_col * sizeof(double));
      if (Z[0][l] && Z[1][l]) {
        for (l = 0; l < layers; l++) {
          r_times_col = rows * nodes[l];
          Z[0][l] = malloc(r_times_col * sizeof(double));
          Z[1][l] = malloc(r_times_col * sizeof(double));
          if (Z[0][l] && Z[1][l]) {
            memset(Z[0][l], 0.0, r_times_col * sizeof(double));
            memset(Z[1][l], 0.0, r_times_col * sizeof(double));
          } else {
            printf(KRED "\nCould not allocate Zs. Aborting...\n" RESET);
            free(Z[0]);
            free(Z[1]);
            free(Z);
            abort();
          }
        }
      } else {
        printf(KRED "\nCould not allocate Zs. Aborting...\n" RESET);
        free(Z[0]);
        free(Z[1]);
        free(Z);
        abort();
      }
    } else {
      printf(KRED "\nCould not allocate Zs. Aborting...\n" RESET);
      free(Z);
      abort();
    }
  } else {
    printf(KRED "\nCould not allocate Zs. Aborting...\n" RESET);
    abort();
  }
  
  functions = name2int(layers, funcs);
  
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
  
  // Directly manipulates Z
  gpu_feedforward_update(&rows, &columns_Y, &columns_X, &layers, Z, X, w, nodes, functions, &context, &device);
  
  memcpy(Z_pred, Z[1][layers], rows * columns_Y * sizeof(double));
  
  free(functions);
  delete_Z(layers, Z);
  return Z_pred;
}
