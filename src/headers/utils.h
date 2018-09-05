#ifndef utils_h__
#define utils_h__
#include <string.h>

extern void activate(double *X_active, double *X, int threshold, char *);
extern void gradient(double *X_graded, double *X, int threshold, char *);
extern void activate_rbf(double *X_active, double *X, double center, double spread, int p, int threshold, char *);
extern void minkowski(double *X, double *Y, int p, int threshold);
extern void randomize(double *X, int rows, int columns_X);
extern void normalize(double *X, int rows, int columns_X);
extern double ***init_wb(double variance, int layers, int nodes[layers], char funcs[layers][30],
                         int columns_Y, int columns_X);
extern double rand_normal(double mu, double sigma);
extern double rmse(int rows, int columns_Y, double *Y, double *Z_active);

// Training utility
extern void row_sum(double *row_sum, double *matrix, int rows, int columns);

// Convolution utility
extern int **im2col(int ***images, int no_of_images, int img_width, int img_height, int img_channels,
                    int spatial, // width and height of weights
                    int stride, // (img_width - spatial + 2 * padding)/stride should be int
                    int padding, // Zeros around
                    int delete_originals // 0 if no, 1 if yes (keep only vectorized in memory)
                  );

// Freedom
extern void delete_wb(int layers, double ***wb);
extern void delete_Z(int layers, double ***Z);
extern void delete_img_vector(int **images, int no_of_images);

#endif  // utils_h__
