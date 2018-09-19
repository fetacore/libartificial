#ifndef neurons_gpu_h__
#define neurons_gpu_h__

#include <stdlib.h>

extern double ***gpu_feedforward(const size_t rows, const size_t columns_Y, const size_t columns_X, const int layers,
                                 const double *X, double ***wb,
                                 const int nodes[layers], char funcs[layers + 1][30]);

#endif  // neurons_gpu_h__
