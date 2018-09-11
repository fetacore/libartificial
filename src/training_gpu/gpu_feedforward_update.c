#include <stdio.h>
#include <stdlib.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS // to disable deprecation warnings
#include "../../clblast/include/clblast_c.h"

#include "../headers/utils.h"

// Updates Z[0 or 1][layer][rows * columns]
// We specify rows as 1 when we do stochastic gd
void feedforward_update_gpu(double ***Z, const size_t rows, const size_t columns_Y, const size_t columns_X,
                            const int layers,
                            const double *X, double ***wb,
                            const int nodes[layers], char funcs[layers + 1][30],
                            const cl_device_id device
                           )
{
  // l is for layers
  // i for each row
  // j for columns at each layer
  int l, for_helper = rows * nodes[0], i = for_helper - 1, j = nodes[0] - 1;
  const double alpha = 1.0;
  const double beta = 0.0;
  
  // Creates the OpenCL context, queue, and an event
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
  cl_event event = NULL;
  
  // Copy the matrices to the device
  cl_mem device_a = clCreateBuffer(context, CL_MEM_READ_WRITE, rows * columns_X * sizeof(double), NULL, NULL);
  cl_mem device_b = clCreateBuffer(context, CL_MEM_READ_WRITE, columns_X * nodes[0] * sizeof(double), NULL, NULL);
  cl_mem device_c = clCreateBuffer(context, CL_MEM_READ_WRITE, rows * nodes[0] * sizeof(double), NULL, NULL);
  clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, rows * columns_X * sizeof(double), X, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, columns_X * nodes[0] * sizeof(double), wb[0][0], 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, device_c, CL_TRUE, 0, rows * nodes[0] * sizeof(double), Z[0][0], 0, NULL, NULL);
  
  CLBlastStatusCode status = CLBlastDgemm(CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeNo,
                                          rows, nodes[0], columns_X,
                                          alpha,
                                          device_a, 0, columns_X,
                                          device_b, 0, nodes[0],
                                          beta,
                                          device_c, 0, nodes[0],
                                          &queue, &event);
  
  // Wait for completion
  clWaitForEvents(1, &event);
  clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, rows * nodes[0] * sizeof(double), Z[0][0], 0, NULL, NULL);
  clReleaseEvent(event);
  
  do {
    if (j < 0) {
      j = nodes[0] - 1;
    }
    Z[0][0][i--] += wb[1][0][j--];
  } while (i >= 0);
  
  activate(Z[1][0], Z[0][0], for_helper, funcs[0]);
  
  // Intermediate layers
  switch (layers > 1) {
    case 0:
      break;
    default:
      l = 1;
      do {
        for_helper = rows * nodes[l];
        i = for_helper - 1;
        j = nodes[l] - 1;
        
        // Copy the matrices to the device
        device_a = clCreateBuffer(context, CL_MEM_READ_WRITE, rows * nodes[l-1] * sizeof(double), NULL, NULL);
        device_b = clCreateBuffer(context, CL_MEM_READ_WRITE, nodes[l-1] * nodes[l] * sizeof(double), NULL, NULL);
        device_c = clCreateBuffer(context, CL_MEM_READ_WRITE, rows * nodes[l] * sizeof(double), NULL, NULL);
        clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, rows * nodes[l-1] * sizeof(double), Z[1][l-1], 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, nodes[l-1] * nodes[l] * sizeof(double), wb[0][l], 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, device_c, CL_TRUE, 0, rows * nodes[l] * sizeof(double), Z[0][l], 0, NULL, NULL);
        
        status = CLBlastDgemm(CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeNo,
                              rows, nodes[l], nodes[l-1],
                              alpha,
                              device_a, 0, nodes[l-1],
                              device_b, 0, nodes[l],
                              beta,
                              device_c, 0, nodes[l],
                              &queue, &event);
        
        // Wait for completion
        clWaitForEvents(1, &event);
        clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, rows * nodes[l] * sizeof(double), Z[0][l], 0, NULL, NULL);
        clReleaseEvent(event);
        
        do {
          if (j < 0) {
            j = nodes[l] - 1;
          }
          Z[0][l][i--] += wb[1][l][j--];
        } while (i >= 0);
        
        activate(Z[1][l], Z[0][l], for_helper, funcs[l]);
        l++;
      } while (l != layers);
      break;
  }
  
  // Output layer
  for_helper = rows * columns_Y;
  i = for_helper - 1;
  j = columns_Y - 1;
  
  // Copy the matrices to the device
  device_a = clCreateBuffer(context, CL_MEM_READ_WRITE, rows * nodes[layers-1] * sizeof(double), NULL, NULL);
  device_b = clCreateBuffer(context, CL_MEM_READ_WRITE, nodes[layers-1] * columns_Y *sizeof(double), NULL, NULL);
  device_c = clCreateBuffer(context, CL_MEM_READ_WRITE, rows * columns_Y * sizeof(double), NULL, NULL);
  clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, rows * nodes[layers-1]*sizeof(double), Z[1][layers-1], 0,NULL,NULL);
  clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, columns_Y * nodes[l] * sizeof(double), wb[0][layers], 0, NULL,NULL);
  clEnqueueWriteBuffer(queue, device_c, CL_TRUE, 0, rows * columns_Y * sizeof(double), Z[0][layers], 0, NULL, NULL);
  
  status = CLBlastDgemm(CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeNo,
                        rows, columns_Y, nodes[layers-1],
                        alpha,
                        device_a, 0, nodes[layers-1],
                        device_b, 0, columns_Y,
                        beta,
                        device_c, 0, columns_Y,
                        &queue, &event);
  
  // Wait for completion
  clWaitForEvents(1, &event);
  clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, rows * columns_Y * sizeof(double), Z[0][layers], 0, NULL, NULL);
  clReleaseEvent(event);
  
  do {
    if (j < 0) {
      j = columns_Y - 1;
    }
    Z[0][layers][i--] += wb[1][layers][j];
  } while (i >= 0);
  
  activate(Z[1][layers], Z[0][layers], for_helper, funcs[layers]);
  
  clReleaseMemObject(device_a);
  clReleaseMemObject(device_b);
  clReleaseMemObject(device_c);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
}
