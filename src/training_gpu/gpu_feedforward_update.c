#include <stdio.h>
#include <stdlib.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS // to disable deprecation warnings
#include "../../clblast/include/clblast_c.h"

#include "../headers/utils.h"

// Updates Z[0 or 1][layer][rows * columns]
// We specify rows as 1 when we do stochastic gd
void gpu_feedforward_update(const size_t rows, const size_t columns_Y, const size_t columns_X, const int layers,
                            double ***Z,
                            double *X, double ***wb,
                            const int nodes[layers], char funcs[layers+1][30],
                            cl_device_id device)
{
  // l is for layers
  // i for each row
  // j for columns at each layer
  int l = 0, i, j, cols, for_helper;
  
  // Creates the OpenCL context, queue, and an event
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
  cl_event event = NULL;
  
  cl_mem device_a, device_b, device_c;
  CLBlastStatusCode status;
  
  do {
    if (l > 0 && l < layers) {
      for_helper = (int)rows * nodes[l];
      cols = nodes[l];
      
      device_a = clCreateBuffer(context, CL_MEM_READ_WRITE, rows * nodes[l-1] * sizeof(double), NULL, NULL);
      device_b = clCreateBuffer(context, CL_MEM_READ_WRITE, nodes[l-1] * nodes[l] * sizeof(double), NULL, NULL);
      device_c = clCreateBuffer(context, CL_MEM_READ_WRITE, rows * nodes[l] * sizeof(double), NULL, NULL);
      
      clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, rows * nodes[l-1] * sizeof(double),
                           Z[1][l-1], 0, NULL, NULL);
      clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, nodes[l-1] * nodes[l] * sizeof(double),
                           wb[0][l], 0, NULL, NULL);
      clEnqueueWriteBuffer(queue, device_c, CL_TRUE, 0, rows * nodes[l] * sizeof(double),
                           Z[0][l], 0, NULL, NULL);
      
      status = CLBlastDgemm(CLBlastLayoutRowMajor, CLBlastTransposeYes, CLBlastTransposeNo,
                            rows, nodes[l], nodes[l-1],
                            1.0,
                            device_a, 0, nodes[l-1],
                            device_b, 0, nodes[l],
                            0.0,
                            device_c, 0, nodes[l],
                            &queue, &event);
      
      // Wait for completion
      if (status == CLBlastSuccess) {
        clWaitForEvents(1, &event);
        clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, rows * nodes[l] * sizeof(double),
                            Z[0][l], 0, NULL, NULL);
        clReleaseEvent(event);
      }
      
    } else if (l == 0) {
      for_helper = (int)rows * nodes[l];
      cols = nodes[l];
      
      device_a = clCreateBuffer(context, CL_MEM_READ_WRITE, rows * columns_X * sizeof(double), NULL, NULL);
      device_b = clCreateBuffer(context, CL_MEM_READ_WRITE, columns_X * nodes[l] * sizeof(double), NULL, NULL);
      device_c = clCreateBuffer(context, CL_MEM_READ_WRITE, rows * nodes[l] * sizeof(double), NULL, NULL);
      
      clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, rows * columns_X * sizeof(double),
                           X, 0, NULL, NULL);
      clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, columns_X * nodes[l] * sizeof(double),
                           wb[0][l], 0, NULL, NULL);
      clEnqueueWriteBuffer(queue, device_c, CL_TRUE, 0, rows * nodes[l] * sizeof(double),
                           Z[0][l], 0, NULL, NULL);
      
      status = CLBlastDgemm(CLBlastLayoutRowMajor, CLBlastTransposeYes, CLBlastTransposeNo,
                            rows, nodes[l], columns_X,
                            1.0,
                            device_a, 0, columns_X,
                            device_b, 0, nodes[l],
                            0.0,
                            device_c, 0, nodes[l],
                            &queue, &event);
      
      // Wait for completion
      if (status == CLBlastSuccess) {
        clWaitForEvents(1, &event);
        clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, rows * nodes[l] * sizeof(double),
                            Z[0][l], 0, NULL, NULL);
        clReleaseEvent(event);
      }
      
    } else {
      for_helper = rows * columns_Y;
      cols = columns_Y;
      
      device_a = clCreateBuffer(context, CL_MEM_READ_WRITE, rows * nodes[l-1] * sizeof(double), NULL, NULL);
      device_b = clCreateBuffer(context, CL_MEM_READ_WRITE, nodes[l-1] * columns_Y * sizeof(double), NULL, NULL);
      device_c = clCreateBuffer(context, CL_MEM_READ_WRITE, rows * columns_Y * sizeof(double), NULL, NULL);
      
      clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, rows * nodes[l-1]* sizeof(double),
                           Z[1][l-1], 0, NULL, NULL);
      clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, nodes[l-1] * columns_Y * sizeof(double),
                           wb[0][l], 0, NULL, NULL);
      clEnqueueWriteBuffer(queue, device_c, CL_TRUE, 0, rows * columns_Y * sizeof(double),
                           Z[0][l], 0, NULL, NULL);
      
      status = CLBlastDgemm(CLBlastLayoutRowMajor, CLBlastTransposeYes, CLBlastTransposeNo,
                            rows, columns_Y, nodes[l-1],
                            1.0,
                            device_a, 0, nodes[l-1],
                            device_b, 0, columns_Y,
                            0.0,
                            device_c, 0, columns_Y,
                            &queue, &event);
      
      // Wait for completion
      if (status == CLBlastSuccess) {
        clWaitForEvents(1, &event);
        clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, rows * columns_Y * sizeof(double),
                            Z[0][l], 0, NULL, NULL);
        clReleaseEvent(event);
      }
    }
    
    j = cols - 1;
    i = for_helper - 1;
    do {
      Z[0][l][i--] += wb[1][l][j--];
      if (j < 0) {
        j = cols - 1;
      }
    } while (i >= 0);
    
    activate(Z[1][l], Z[0][l], for_helper, funcs[l]);
    l++;
  } while (l < layers + 1);
  clReleaseMemObject(device_a);
  clReleaseMemObject(device_b);
  clReleaseMemObject(device_c);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
}
