#ifndef training_gpu_h__
#define training_gpu_h__

#include <stdlib.h>
#include "../../clblast/include/clblast_c.h"

// While updating we need to find Z with the new weights and biases
extern void feedforward_update_gpu(double ***Z, const size_t rows, const size_t columns_Y, const size_t columns_X,
                                   const int layers,
                                   const double *X, double ***wb,
                                   const int nodes[layers], char funcs[layers+1][30],
                                   const cl_device_id device);

extern void update_gd_gpu(const size_t rows, const size_t columns_Y, const size_t columns_X,
                          const int batch, const int layers, const int nodes[layers],
                          const double *Y, const double *X, double ***Z, double ***wb,
                          char funcs[layers+1][30], const double learning_rate, const int epochs);

extern void delta_gd_gpu(double **deltas, const size_t rows, const size_t columns_Y, const int layers,
                         const double *Y_row, double ***Z_row, double ***wb,
                         const int nodes[layers], char funcs[layers+1][30],
                         const cl_device_id device);

#endif  // training_gpu_h__
