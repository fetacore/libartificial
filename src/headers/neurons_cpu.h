#ifndef neurons_cpu_h__
#define neurons_cpu_h__

#include <stdlib.h>

extern double ***cpu_feedforward(const int rows, const int columns_Y, const int columns_X, const int layers,
                                 double *X, double ***wb,
                                 const int nodes[layers], char funcs[layers+1][30]);

#endif  // neurons_cpu_h__
