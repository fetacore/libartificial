#ifndef cpu_h__
#define cpu_h__

// While updating we need to find Z with the new weights and biases

extern void cpu_feedforward_update(const int *restrict rows, const int *restrict columns_Y,
                                   const int *restrict columns_X, const int *restrict layers,
                                   double ***restrict Z,
                                   const double *restrict X, double **restrict weights,
                                   const int *restrict nodes, const int *restrict funcs);

extern double ***cpu_feedforward_cache(const int *restrict rows, const int *restrict columns_Y,
                                       const int *restrict columns_X, const int *restrict layers,
                                       const double *restrict X, double **restrict weights,
                                       const int *restrict nodes, const int *restrict funcs);

extern void cpu_gd_delta(double **restrict deltas, double **restrict help_1, double **restrict help_2,
                         const int *restrict rows, const int *restrict columns_Y, const int *restrict layers,
                         const double *restrict Y_row, double ***restrict Z_row, double **restrict weights,
                         const int *restrict nodes, const int *restrict funcs);

#endif // cpu_h__
