#ifndef training_cpu_h__
#define training_cpu_h__

#include <stdlib.h>

// While updating we need to find Z with the new weights and biases
extern void cpu_feedforward_update(const size_t rows, const size_t columns_Y, const size_t columns_X, const int layers,
                                   double ***Z,
                                   double *X, double ***wb,
                                   const int nodes[layers], char funcs[layers+1][30]);

extern void cpu_gd_update(const size_t rows, const size_t columns_Y, const size_t columns_X,
                          const int batch, const int layers,
                          const int nodes[layers],
                          double *Y, double *X, double ***Z, double ***wb,
                          char funcs[layers+1][30], const double learning_rate, const int epochs);

extern void cpu_gd_delta(double **deltas,
                         const size_t rows, const size_t columns_Y, const int layers,
                         double *Y_row, double ***Z_row, double ***wb,
                         const int nodes[layers], char funcs[layers+1][30]);

#endif  // training_cpu_h__
