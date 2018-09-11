#include <stdio.h>
#include <stdlib.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS // to disable deprecation warnings
#include "../../clblast/include/clblast_c.h"

#include "../headers/utils.h"
#include "../headers/training_gpu.h"

double ***feedforward_gpu(const size_t rows, const size_t columns_Y, const size_t columns_X, const int layers,
                          const double *X, double ***wb,
                          const int nodes[layers], char funcs[layers+1][30])
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
  
  // Directly manipulates Z
  feedforward_update_gpu(Z, rows, columns_Y, columns_X, layers, X, wb, nodes, funcs, device);
  
  free(platforms);
  free(devices);
  
  return Z;
}
