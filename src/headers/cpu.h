#ifndef cpu_h__
#define cpu_h__

// While updating we need to find Z with the new weights and biases

extern void cpu_feedforward_update(const int *rows, const int *columns_Y, const int *columns_X, const int *layers,
                                   double ***Z,
                                   const double *X, double **weights,
                                   const int *nodes, const int *funcs);

extern double ***cpu_feedforward_cache(const int *rows, const int *columns_Y, const int *columns_X, const int *layers,
                                       const double *X, double **weights,
                                       const int *nodes, const int *funcs);

extern void cpu_gd_delta(double **deltas,
                         const int *rows, const int *columns_Y, const int *layers,
                         const double *Y_row, double ***Z_row, double **weights,
                         const int *nodes, const int *funcs);

#endif // cpu_h__
