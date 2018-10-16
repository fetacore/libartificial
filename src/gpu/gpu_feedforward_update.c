#include <stdio.h>
#include <stdlib.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS // to disable deprecation warnings
#include "../../clblast/include/clblast_c.h"

#include "../headers/utils.h"

// Updates Z[0 or 1][layer][rows * columns]
// We specify rows as 1 when we do stochastic gd
void gpu_feedforward_update(const int *rows, const int *columns_Y, const int *columns_X, const int *layers,
                            double ***Z,
                            double *X, double **w,
                            const int *nodes, const int *funcs,
                            cl_context *context, cl_device_id *device)
{
  // l is for layers
  int l = 0, for_helper = (*rows) * nodes[l];
  cl_command_queue queue = clCreateCommandQueue((*context), (*device), 0, NULL);
  cl_event event = NULL;
  
  cl_mem dev_a, dev_b, dev_c;
  CLBlastStatusCode status;
  
  do {
    switch (l == 0) {
      case 1:
        for_helper = (*rows) * nodes[l];
        
        dev_a = clCreateBuffer((*context), CL_MEM_READ_WRITE, (*rows) * (*columns_X) * sizeof(double), NULL, NULL);
        dev_b = clCreateBuffer((*context), CL_MEM_READ_WRITE, (*columns_X) * nodes[l] * sizeof(double), NULL, NULL);
        dev_c = clCreateBuffer((*context), CL_MEM_READ_WRITE, (*rows) * nodes[l] * sizeof(double), NULL, NULL);
        
        clEnqueueWriteBuffer(queue, dev_a, CL_TRUE, 0, (*rows) * (*columns_X) * sizeof(double),
                             X, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, dev_b, CL_TRUE, 0, (*columns_X) * nodes[l] * sizeof(double),
                             w[l], 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, dev_c, CL_TRUE, 0, (*rows) * nodes[l] * sizeof(double),
                             Z[0][l], 0, NULL, NULL);
        
        status = CLBlastDgemm(CLBlastLayoutRowMajor, CLBlastTransposeYes, CLBlastTransposeNo,
                              (*rows), nodes[l], (*columns_X),
                              1.0,
                              dev_a, 0, (*columns_X),
                              dev_b, 0, nodes[l],
                              0.0,
                              dev_c, 0, nodes[l],
                              &queue, &event);
        
        // Wait for completion
        if (status == CLBlastSuccess) {
          clWaitForEvents(1, &event);
          clEnqueueReadBuffer(queue, dev_c, CL_TRUE, 0, (*rows) * nodes[l] * sizeof(double),
                              Z[0][l], 0, NULL, NULL);
          clReleaseEvent(event);
          clReleaseMemObject(dev_a);
          clReleaseMemObject(dev_b);
          clReleaseMemObject(dev_c);
        }
        break;
      default:
        break;
    }
    
    switch (l > 0 && l < (*layers)) {
      case 1:
        for_helper = (*rows) * nodes[l];
        
        dev_a = clCreateBuffer((*context), CL_MEM_READ_WRITE, (*rows) * nodes[l-1] * sizeof(double), NULL, NULL);
        dev_b = clCreateBuffer((*context), CL_MEM_READ_WRITE, nodes[l-1] * nodes[l] * sizeof(double), NULL, NULL);
        dev_c = clCreateBuffer((*context), CL_MEM_READ_WRITE, (*rows) * nodes[l] * sizeof(double), NULL, NULL);
        
        clEnqueueWriteBuffer(queue, dev_a, CL_TRUE, 0, (*rows) * nodes[l-1] * sizeof(double),
                             Z[1][l-1], 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, dev_b, CL_TRUE, 0, nodes[l-1] * nodes[l] * sizeof(double),
                             w[l], 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, dev_c, CL_TRUE, 0, (*rows) * nodes[l] * sizeof(double),
                             Z[0][l], 0, NULL, NULL);
        
        status = CLBlastDgemm(CLBlastLayoutRowMajor, CLBlastTransposeYes, CLBlastTransposeNo,
                              (*rows), nodes[l], nodes[l-1],
                              1.0,
                              dev_a, 0, nodes[l-1],
                              dev_b, 0, nodes[l],
                              0.0,
                              dev_c, 0, nodes[l],
                              &queue, &event);
        
        // Wait for completion
        if (status == CLBlastSuccess) {
          clWaitForEvents(1, &event);
          clEnqueueReadBuffer(queue, dev_c, CL_TRUE, 0, (*rows) * nodes[l] * sizeof(double),
                              Z[0][l], 0, NULL, NULL);
          clReleaseEvent(event);
          clReleaseMemObject(dev_a);
          clReleaseMemObject(dev_b);
          clReleaseMemObject(dev_c);
        }
        break;
      default:
        break;
    }
    
    switch (l == (*layers)) {
      case 1:
        for_helper = (*rows) * (*columns_Y);
        
        dev_a = clCreateBuffer((*context), CL_MEM_READ_WRITE, (*rows) * nodes[l-1] * sizeof(double), NULL, NULL);
        dev_b = clCreateBuffer((*context), CL_MEM_READ_WRITE, nodes[l-1] * (*columns_Y) * sizeof(double), NULL, NULL);
        dev_c = clCreateBuffer((*context), CL_MEM_READ_WRITE, (*rows) * (*columns_Y) * sizeof(double), NULL, NULL);
        
        clEnqueueWriteBuffer(queue, dev_a, CL_TRUE, 0, (*rows) * nodes[l-1]* sizeof(double),
                             Z[1][l-1], 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, dev_b, CL_TRUE, 0, nodes[l-1] * (*columns_Y) * sizeof(double),
                             w[l], 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, dev_c, CL_TRUE, 0, (*rows) * (*columns_Y) * sizeof(double),
                             Z[0][l], 0, NULL, NULL);
        
        status = CLBlastDgemm(CLBlastLayoutRowMajor, CLBlastTransposeYes, CLBlastTransposeNo,
                              (*rows), (*columns_Y), nodes[l-1],
                              1.0,
                              dev_a, 0, nodes[l-1],
                              dev_b, 0, (*columns_Y),
                              0.0,
                              dev_c, 0, (*columns_Y),
                              &queue, &event);
        
        // Wait for completion
        if (status == CLBlastSuccess) {
          clWaitForEvents(1, &event);
          clEnqueueReadBuffer(queue, dev_c, CL_TRUE, 0, (*rows) * (*columns_Y) * sizeof(double),
                              Z[0][l], 0, NULL, NULL);
          clReleaseEvent(event);
          clReleaseMemObject(dev_a);
          clReleaseMemObject(dev_b);
          clReleaseMemObject(dev_c);
        }
        break;
      default:
        break;
    }
    activate(Z[1][l], Z[0][l], &for_helper, &funcs[l]);
  } while (++l < (*layers) + 1);
  clReleaseCommandQueue(queue);
}

