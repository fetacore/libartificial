#ifndef utils_h__
#define utils_h__

#include <stdlib.h>
#include <string.h>

extern void activate(double *X_active, double *X, const int threshold, char *);
extern void gradient(double *X_graded, double *X, const int threshold, char *);
extern void randomize(double *X, const size_t rows, const size_t columns_X);
extern void normalize(double *X, const size_t rows, const size_t columns_X);
extern double ***init_wb(const double variance, const int layers, const int nodes[layers], char funcs[layers+1][30],
                         const size_t columns_Y, const size_t columns_X);
extern double rand_normal(const double mu, const double sigma);
extern double rmse(const size_t rows, const size_t columns_Y, const double *Y, const double *Z_active);

// Training utility
extern void row_sum(double *row_sum, double *matrix, const size_t rows, const size_t columns);

// Convolution utility
extern int **im2col(int ***images,const int no_of_images,
                    const size_t img_width, const size_t img_height, const size_t img_channels,
                    const size_t spatial, // width and height of weights
                    const size_t stride, // (img_width - spatial + 2 * padding)/stride should be int
                    const size_t padding, // Zeros around
                    const size_t delete_originals // 0 if no, 1 if yes (keep only vectorized in memory)
                   );

// Saving and loading wb files
extern void save_wb(double ***wb, const int layers, const int nodes[layers],
                    const size_t columns_Y, const size_t columns_X);
extern double ***load_wb(const int layers, const int nodes[layers], const size_t columns_Y, const size_t columns_X);

// Freedom
extern void delete_wb(const int layers, double ***wb);
extern void delete_Z(const int layers, double ***Z);
extern void delete_img_vector(int **images, const size_t no_of_images);

#endif  // utils_h__
