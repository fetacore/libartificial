#ifndef utils_h__
#define utils_h__
#include <string.h>

extern void activate(double *X_active, double *X, int threshold, char *);
extern void gradient(double *X_graded, double *X, int threshold, char *);
extern void activate_rbf(double *X_active, double *X,
												 double center, double spread,
												 int p, int threshold, char *);
extern void minkowski(double *X, double *Y, int p, int threshold);
extern void randomize(double *X, int rows, int columns_X);
extern void normalize(double *X, int rows, int columns_X);
extern double ***init_wb(double variance, 
												 int layers, int nodes[layers], char funcs[layers][30],
												 int columns_Y, int columns_X);
extern double rand_normal(double mu, double sigma);
extern double rmse(int rows, int columns_Y, double *Y, double *Z_active);

// Training utility
extern void row_sum(double *row_sum, double *matrix, int rows, int columns);

// Freedom
extern void delete_wb(int layers, double ***wb);
extern void delete_Z(int layers, double ***Z);

#endif  // utils_h__
