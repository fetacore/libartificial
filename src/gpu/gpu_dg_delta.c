#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS // to disable deprecation warnings
#include "../../clblast/include/clblast_c.h"

#include "../headers/utils.h"

#define KRED  "\x1B[31m"
#define RESET "\033[0m"

// DELTAS FOR EVERY TYPE OF GRADIENT DESCENT
// (batch => rows = all rows)
// (stochastic => rows = 1)
// (mini-batch => rows = rows of mini set)

void gpu_gd_delta(double **deltas,
                  const int *rows, const int *columns_Y, const int *layers,
                  double *Y, double ***Z, const double **w,
                  const int *nodes, const int *funcs,
                  cl_context *context, cl_device_id *device)
{
  int l, i = (*rows) * (*columns_Y) - 1;
  
  //////////////////////////////////////////////////////////////////
  // Gradient of layer's unactivated output
  double **help_1 = malloc((*layers) * sizeof(double *));
  // Product of next layer's transposed weights and deltas
  double **help_2 = malloc((*layers) * sizeof(double *));
  // We do not need them at the output layer
  //////////////////////////////////////////////////////////////////
  if (help_1 && help_2) {
    for (l = 0; l < (*layers); l++) {
      help_1[l] = malloc((*rows) * nodes[l] * sizeof(double));
      help_2[l] = malloc((*rows) * nodes[l] * sizeof(double));
      if (help_1[l]) {
        memset(help_1[l], 0.0, (*rows) * nodes[l] * sizeof(double));
        memset(help_2[l], 0.0, (*rows) * nodes[l] * sizeof(double));
      } else {
        printf(KRED "\nFailed to allocate helpers for delta. Aborting...\n" RESET);
        free(help_1);
        free(help_2);
        return;
      }
    }
  } else {
    printf(KRED "\nFailed to allocate helpers for delta. Aborting...\n" RESET);
    return;
  }
  
  // Last layer
  switch (funcs[(*layers)]) {
    case 2:
      do {
        deltas[(*layers)][i] = Z[1][(*layers)][i] - Y[i];
      } while (--i >= 0);
      break;
    default:
      gradient(deltas[(*layers)], Z[0][(*layers)], &i, &funcs[(*layers)]);
      do {
        deltas[(*layers)][i] = (Z[1][(*layers)][i] - Y[i]) * deltas[(*layers)][i];
      } while (--i >= 0);
      break;
  }
  
  cl_command_queue queue = clCreateCommandQueue((*context), (*device), 0, NULL);
  cl_event event = NULL;
  
  // Before last
  l = (*layers) - 1;
  i = (*rows) * nodes[l] - 1;
  
  gradient(help_1[l], Z[0][l], &i, &funcs[l]);
  
  cl_mem device_a = clCreateBuffer((*context), CL_MEM_READ_WRITE, (*rows) * (*columns_Y) * sizeof(double), NULL, NULL);
  cl_mem device_b = clCreateBuffer((*context), CL_MEM_READ_WRITE, nodes[l] * (*columns_Y) * sizeof(double), NULL, NULL);
  cl_mem device_c = clCreateBuffer((*context), CL_MEM_READ_WRITE, (*rows) * nodes[l] * sizeof(double), NULL, NULL);
  
  clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, (*rows) * (*columns_Y) * sizeof(double),
                       deltas[l+1], 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, nodes[l] * (*columns_Y) * sizeof(double),
                       w[l+1], 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, device_c, CL_TRUE, 0, (*rows) * nodes[l] * sizeof(double),
                       help_2[l], 0, NULL, NULL);
  
  CLBlastStatusCode status = CLBlastDgemm(CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeYes,
                                          (*rows), nodes[l], (*columns_Y),
                                          1.0,
                                          device_a, 0, (*columns_Y),
                                          device_b, 0, (*columns_Y),
                                          0.0,
                                          device_c, 0, nodes[l],
                                          &queue, &event);
  
  // Wait for completion
  if (status == CLBlastSuccess) {
    clWaitForEvents(1, &event);
    clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, (*rows) * nodes[l] * sizeof(double),
                        help_2[l], 0, NULL, NULL);
    clReleaseEvent(event);
    clReleaseMemObject(device_a);
    clReleaseMemObject(device_b);
    clReleaseMemObject(device_c);
  }
  
  // Hadamard product
  do {
    deltas[l][i] = help_1[l][i] * help_2[l][i];
  } while (--i >= 0);
  
  switch ((*layers) == 1) {
    case 1:
      clReleaseCommandQueue(queue);
      for (l = 0; l < (*layers); l++) {
        free(help_2[l]);
        free(help_1[l]);
      }
      free(help_2);
      free(help_1);
      return;
    default:
      // All other layers
      l = (*layers) - 2;
      do {
        i = (*rows) * nodes[l] - 1;
        
        gradient(help_1[l], Z[0][l], &i, &funcs[l]);
        
        device_a = clCreateBuffer((*context), CL_MEM_READ_WRITE, (*rows) * nodes[l+1] * sizeof(double), NULL, NULL);
        device_b = clCreateBuffer((*context), CL_MEM_READ_WRITE, nodes[l] * nodes[l+1] * sizeof(double), NULL, NULL);
        device_c = clCreateBuffer((*context), CL_MEM_READ_WRITE, (*rows) * nodes[l] * sizeof(double), NULL, NULL);
        
        clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, (*rows) * nodes[l+1] * sizeof(double),
                             deltas[l+1], 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, nodes[l] * nodes[l+1] * sizeof(double),
                             w[l+1], 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, device_c, CL_TRUE, 0, (*rows) * nodes[l] * sizeof(double),
                             help_2[l], 0, NULL, NULL);
        
        status = CLBlastDgemm(CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeYes,
                              (*rows), nodes[l], (*columns_Y),
                              1.0,
                              device_a, 0, (*columns_Y),
                              device_b, 0, (*columns_Y),
                              0.0,
                              device_c, 0, nodes[l],
                              &queue, &event);
        
        // Wait for completion
        if (status == CLBlastSuccess) {
          clWaitForEvents(1, &event);
          clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, (*rows) * nodes[l] * sizeof(double),
                              help_2[l], 0, NULL, NULL);
          clReleaseEvent(event);
          clReleaseMemObject(device_a);
          clReleaseMemObject(device_b);
          clReleaseMemObject(device_c);
        }
        
        // Hadamard product
        do {
          deltas[l][i] = help_1[l][i] * help_2[l][i];
        } while (--i >= 0);
      } while (--l >= 0);
      clReleaseCommandQueue(queue);
      for (l = 0; l < (*layers); l++) {
        free(help_2[l]);
        free(help_1[l]);
      }
      free(help_2);
      free(help_1);
      return;
  }
}

