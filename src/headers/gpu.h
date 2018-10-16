#ifndef gpu_h__
#define gpu_h__

#ifdef __APPLE__
  #include "OpenCL/opencl.h"
#else
  #include "CL/cl.h"
#endif

extern double ***gpu_feedforward_cache(const int *rows, const int *columns_Y, const int *columns_X, const int *layers,
                                       double *X, double **weights,
                                       const int *nodes, const int *funcs,
                                       cl_context *context, cl_device_id *device);

// While updating we need to find Z with the new weights and biases
extern void gpu_feedforward_update(const int *rows, const int *columns_Y, const int *columns_X, const int *layers,
                                   double ***Z,
                                   double *X, double **weights,
                                   const int *nodes, const int *funcs,
                                   cl_context *context, cl_device_id *device);


extern void gpu_gd_delta(double **deltas,
                        const int *rows, const int *columns_Y, const int *layers,
                        double *Y_row, double ***Z_row, double **weights,
                        const int *nodes, const int *funcs,
                        cl_context *context, cl_device_id *device);

#endif // gpu_h__
