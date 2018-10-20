/*
 * libartificial - Small header-only C library for Artificial Neural Networks
 * 
 * Copyright (c) 2018 Jim Karoukis
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */


#ifndef libartificial_h__
#define libartificial_h__

#ifdef __WIN32__
  #if defined(COMPILING_DLL)
    #define PUBLIC_API __declspec(dllexport)
  #else
    #define PUBLIC_API __declspec(dllimport)
  #endif
#else
  #define PUBLIC_API
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include "./OpenBLAS/cblas.h"

#define KRED "\x1B[31m"
#define KGRN  "\x1B[32m"
#define RESET "\033[0m"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Forward declarations
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// UTILITIES ///////////////////////////////////////////////////////////////////////////////////////////////////////////
// For randomization
static inline void swap(double *restrict a, double *restrict b);
// For normalization
static inline double mean(const int *restrict rows, const double *restrict col);
static inline double stdev(const int *restrict rows, const double *restrict col, const double *restrict mean);
// To convert array of names into array of ints for fast gradient and activations
static inline int *name2int(const int *restrict layers, char funcs[(*layers) + 1][30]);
// Activations
static inline void activate(double *restrict Y, const double *restrict X,
                            const int *restrict r, const int *restrict c,
                            const int *restrict f);
// Gradients
static inline void gradient(double *restrict Y, const double *restrict X,
                            const int *restrict r, const int *restrict c,
                            const int *restrict f);
// Losses
static inline double rmse(const int *restrict r, const int *restrict c,
                          const double *restrict Y, const double *restrict Z);
static inline double xentropy(const int *restrict r, const int *restrict c,
                              const double *restrict Y, const double *restrict Z);
// Store already trained weights (used by cpu_gd_train)
static inline void save_w(double **restrict weights, const int *restrict layers, const int *restrict nodes,
                          const int *restrict cols_Y, const int *restrict cols_X);
// Convolution utilities
static inline int ***imgpad(int ***restrict images, const int *restrict no_of_images,
                            const int *restrict img_width,
                            const int *restrict img_height,
                            const int *restrict img_channels,
                            const int *restrict padding, // Zeros around
                            const int *restrict delete_originals); // 0 = no, 1 = yes (keep only vector in memory)

static inline int **im2col(int ***restrict images,const int *restrict no_of_images,
                           const int *restrict img_width,
                           const int *restrict img_height,
                           const int *restrict img_channels,
                           const int *restrict spatial, // width and height of weights
                           const int *restrict stride, // (img_width - spatial + 2 * padding)/stride should be int
                           const int *restrict padding, // Zeros around
                           const int *restrict delete_originals); // 0 = no, 1 = yes (keep only vectorized in memory)
// Freedom
static inline void delete_Z(const int *restrict layers, double ***restrict Z);
static inline void delete_img_vector(const int *restrict no_of_images, int **restrict images);
// End of UTILITIES ////////////////////////////////////////////////////////////////////////////////////////////////////

// Training Utilities
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Feedforward pass
static inline void cpu_feedforward_update(const int *restrict r, const int *restrict cY,
                                          const int *restrict cX, const int *restrict layers,
                                          double ***restrict Z,
                                          const double *restrict X, double **restrict w,
                                          const int *restrict n, const int *restrict f);

static inline double ***cpu_feedforward_cache(const int *restrict r, const int *restrict cY,
                                              const int *restrict cX, const int *restrict layers,
                                              const double *restrict X, double **restrict w,
                                              const int *restrict n, const int *restrict f);
// Deltas
static inline void cpu_gd_delta(double **restrict d, double **restrict h1, double **restrict h2,
                                const int *restrict r, const int *restrict c, const int *restrict layers,
                                const double *restrict Y, double ***restrict Z, double **restrict w,
                                const int *restrict n, const int *restrict f);

// returns new wb
static inline void cpu_threaded_update(const double *restrict X, const double *restrict d,
                                       double *restrict gw,
                                       double *restrict w,
                                       const int *restrict m,
                                       const int *restrict n,
                                       const int *restrict k,
                                       const double *restrict c);

// End of forward declarations
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// PUBLIC API

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static inline void PUBLIC_API randomize(double *restrict X, const int *restrict rows, const int *restrict cols)
{
  // Use a different seed value so that we don't get same
  // result each time we run this program
  srand(time(NULL));
  // Start from the last element and swap one by one. We don't
  // need to run for the first element that's why --i > 0
  int i = (*rows) * (*cols) - 1;
  do {
    // Pick a random index from 0 to i
    int j = rand() % (i+1);
    // Swap X[i] with the element at random index
    swap(&X[i], &X[j]);
  } while (--i > 0);
}

static inline void PUBLIC_API normalize(double *restrict X, const int *restrict rows, const int *restrict cols)
{
  int i, j;
  double m = 0.0, sd = 0.0, col[(*rows)];
  for (j = 0; j < (*cols); j++) {
    if (j != 0) {
      for (i = 0; i < (*rows); i++) {
        col[i] = X[i * (*cols) + j];
      }
      m = mean(rows, col);
      sd = stdev(rows, col, &m);
      i = (*rows) - 1;
      for (i = 0; i < (*rows); i++) {
        X[i * (*cols) + j] = (X[i * (*cols) + j] - m)/sd;
      }
      m = 0.0;
      sd = 0.0;
    }
  }
}

static inline double PUBLIC_API rand_normal(const double mu, const double sigma)
{
  static double n2 = 0.0;
  static double n2_cached = 0.0;
  if (!n2_cached) {
    double x, y, r;
    do
    {
      x = 2.0 * (double)rand()/RAND_MAX - 1;
      y = 2.0 * (double)rand()/RAND_MAX - 1;
      
      r = x*x + y*y;
    } while (r == 0.0 || r > 1.0);
    double d = sqrt(-2.0 * log(r)/r);
    double n1 = x * d;
    n2 = y*d;
    double result = n1 * sigma + mu;
    n2_cached = 1.0;
    return result;
  } else {
    n2_cached = 0.0;
    return n2 * sigma + mu;
  }
}

//	Variance is needed since depending on the data, tanh/relu may give nans.
//	Variance < 1 and close to 0.01 if data range too large
static inline PUBLIC_API double **init_w(const double *restrict variance, const int *restrict layers,
                                         const int *restrict nodes, char funcs[(*layers)][30],
                                         const int *restrict cols_Y, const int *restrict cols_X)
{
  // l layers
  int l = (*layers), prod;
  // wb[0] is weights;
  // wb[1] is biases;
  double **weights = malloc((l + 1) * sizeof(double *));
  // For the heuristics of weight initialization
  double correction;
  do {
    int isRelu = strcmp(funcs[l], "relu") + strcmp(funcs[l], "lrelu");
    int isTanh = strcmp(funcs[l], "tanh");
    switch (l > 0 && l < (*layers)) {
      // The statement is false
      case 0:
        switch (l == 0) {
          // The statement is true
          case 1:
            prod = (*cols_X) * nodes[l];
            weights[l] = malloc(prod * sizeof(double));
            switch (isRelu == 2 && isTanh == 1) {
              // One of the two is false
              case 0:
                switch (isRelu == 1) {
                  // Either relu or lrelu
                  case 1:
                    // He et al.
                    correction = sqrt(2.0/(double)(*cols_X));
                    break;
                  default:
                    // Xavier
                    correction = sqrt(1.0/(double)(*cols_X));
                    break;
                }
                break;
              default:
                correction = sqrt(2.0/(double)(prod));
                break;
            }
            --prod;
            do {
              weights[l][prod] = correction * rand_normal(0.0, (*variance));
            } while (--prod >= 0);
            break;
          // l = layers
          default:
            prod = nodes[l-1] * (*cols_Y);
            weights[l] = malloc(prod * sizeof(double));
            switch (isRelu == 2 && isTanh == 1) {
              case 0:
                switch (isRelu == 1) {
                  // Either relu or lrelu
                  case 1:
                    // He et al.
                    correction = sqrt(2.0/(double)nodes[l-1]);
                    break;
                  default:
                    // Xavier
                    correction = sqrt(1.0/(double)nodes[l-1]);
                    break;
                }
                break;
              default:
                correction = sqrt(2.0/(double)(prod));
                break;
            }
            --prod;
            do {
              weights[l][prod] = correction * rand_normal(0.0, (*variance));
            } while (--prod >= 0);
            break;
        }
        break;
      // We are in between input and output
      default:
        prod = nodes[l-1] * nodes[l];
        weights[l] = malloc(prod * sizeof(double));
        
        switch (isRelu == 2 && isTanh == 1) {
          case 0:
            switch (isRelu == 1) {
              // Either relu or lrelu
              case 1:
                // He et al.
                correction = sqrt(2.0/(double)nodes[l-1]);
                break;
              default:
                // Xavier
                correction = sqrt(1.0/(double)nodes[l-1]);
                break;
            }
            break;
          default:
            correction = sqrt(2.0/(double)(prod));
            break;
        }
        --prod;
        do {
          weights[l][prod] = correction * rand_normal(0.0, (*variance));
        } while (--prod >= 0);
        break;
    }
  } while (--l >= 0);
  printf(KGRN "\nWeights and biases initialized successfully!\n" RESET);
  return weights;
}

// Load pretrained wb files
static inline double PUBLIC_API **load_w(const int *restrict layers, const int *restrict nodes,
                                         const int *restrict cols_Y, const int *restrict cols_X)
{
  int l;
  char path[1024];
  getcwd(path, sizeof(path));
  char w_path[1036];
  strcpy(w_path, path);
  strcat(w_path, "/weights");
  double **w = malloc(((*layers) + 1) * sizeof(double *));
  w[0] = malloc((*cols_X) * nodes[0] * sizeof(double));  
  switch ((*layers) > 1) {
    case 1:
      for (l = 1; l < (*layers); l++) {
        w[l] = malloc(nodes[l-1] * nodes[l] * sizeof(double));
      }
      break;
    default:
      break;
  }
  w[(*layers)] = malloc(nodes[(*layers) - 1] * (*cols_Y) * sizeof(double));
  FILE *ptr_fp;
  for (l = 0; l < (*layers) + 1; l++) {
    // This needs to change every time
    char w_path_filename[1050];
    char number[100];
    char filename[15] = "layer_";
    sprintf(number, "%d", l);
    strcat(filename, number);
    strcat(filename, ".bin");
    
    strcpy(w_path_filename, w_path);
    strcat(w_path_filename, "/");
    strcat(w_path_filename, filename);
    
    if((ptr_fp = fopen(w_path_filename, "rb")) == NULL) {
      printf("Unable to open file!\n");
      exit(1);
    }
    
    if (l == 0) {
      if(fread(w[l], (*cols_X) * nodes[l] * sizeof(double), 1, ptr_fp) != 1) {
        printf("Read error!\n");
        exit(1);
      }
      fclose(ptr_fp);
    } else if (l == (*layers)) {
      if(fread(w[l], nodes[l-1] * (*cols_Y) * sizeof(double), 1, ptr_fp) != 1) {
        printf("Read error!\n");
        exit(1);
      }
      fclose(ptr_fp);
    } else {
      if(fread(w[l], nodes[l-1] * nodes[l] * sizeof(double), 1, ptr_fp) != 1) {
        printf("Read error!\n");
        exit(1);
      }
      fclose(ptr_fp);
    }
  }
  return w;
}

// Free wb
static inline void PUBLIC_API delete_w(const int *restrict layers, double **restrict w)
{
  int l;
  for (l = 0; l < (*layers) + 1; l++) free(w[l]);
  free(w);
}

// Actual perceptron
static inline double PUBLIC_API *cpu_feedforward_predict(const int *restrict rows, const int *restrict cols_Y,
                                                         const int *restrict cols_X, const int *restrict layers,
                                                         const double *restrict X, double **restrict w,
                                                         const int *restrict nodes, char funcs[(*layers) + 1][30])
{
  // l is for layers
  // i is for each row * column of X, Y
  int l = (*layers);
  int r_times_col = (*rows) * (*cols_Y);
  int *f;
  
  double *Z_pred = malloc(r_times_col * sizeof(double));
  if (Z_pred) {
    memset(Z_pred, 0.0, r_times_col * sizeof(double));
  } else {
    printf(KRED "\nCould not allocate Z_prediction. Aborting...\n" RESET);
    abort();
  }
  
  // feeds at every layer
  double ***Z = malloc(2 * sizeof(double **));
  if (Z) {
    Z[0] = malloc(((*layers) + 1) * sizeof(double *));
    Z[1] = malloc(((*layers) + 1) * sizeof(double *));
    if (Z[0] && Z[1]) {
      Z[0][l] = malloc(r_times_col * sizeof(double));
      Z[1][l] = malloc(r_times_col * sizeof(double));
      if (Z[0][l] && Z[1][l]) {
        for (l = 0; l < (*layers); l++) {
          r_times_col = (*rows) * nodes[l];
          Z[0][l] = malloc(r_times_col * sizeof(double));
          Z[1][l] = malloc(r_times_col * sizeof(double));
          if (Z[0][l] && Z[1][l]) {
            memset(Z[0][l], 0.0, r_times_col * sizeof(double));
            memset(Z[1][l], 0.0, r_times_col * sizeof(double));
          } else {
            printf(KRED "\nCould not allocate Zs. Aborting...\n" RESET);
            free(Z[0]);
            free(Z[1]);
            free(Z);
            free(Z_pred);
            abort();
          }
        }
      } else {
        printf(KRED "\nCould not allocate Zs. Aborting...\n" RESET);
        free(Z[0]);
        free(Z[1]);
        free(Z);
        free(Z_pred);
        abort();
      }
    } else {
      printf(KRED "\nCould not allocate Zs. Aborting...\n" RESET);
      free(Z);
      free(Z_pred);
      abort();
    }
  } else {
    printf(KRED "\nCould not allocate Zs. Aborting...\n" RESET);
    free(Z_pred);
    abort();
  }
  
  f = name2int(layers, funcs);
  
  // Directly manipulates Z
  cpu_feedforward_update(rows, cols_Y, cols_X, layers, Z, X, w, nodes, f);
  
  memcpy(Z_pred, Z[1][(*layers)], (*rows) * (*cols_Y) * sizeof(double));
  
  free(f);
  delete_Z(layers, Z);
  return Z_pred;
}

static inline void PUBLIC_API cpu_gd_train(const int *restrict rows,
                                           const int *restrict cols_Y,
                                           const int *restrict cols_X,
                                           const int *restrict batch,
                                           const int *restrict layers,
                                           const int *restrict nodes,
                                           const double *restrict Y,
                                           const double *restrict X,
                                           double **restrict w,
                                           char funcs[(*layers) + 1][30],
                                           const double *restrict learning_rate,
                                           const int *restrict epochs)
{
  // l for layers
  // i for rows
  // e for epochs
  int l, i, for_helper_w, for_helper_batch, e = (*epochs), r_over_b = (*rows)/(*batch);
  register int *f;
  f = name2int(layers, funcs);
  
  // Multiplication in threads
  openblas_set_num_threads(1);
  goto_set_num_threads(1);
  
  double loss = 0.0;
  
  // For the averaging of deltas in batch/mini-batch
  double correction = 1.0;
  
  register double **deltas = malloc(((*layers) + 1) * sizeof(double *));
  // The values to be subtracted from weights
  register double **grad_w = malloc(((*layers) + 1) * sizeof(double *));
  //////////////////////////////////////////////////////////////////
  // Gradient of layer's unactivated output
  double **help_1 = malloc((*layers) * sizeof(double *));
  // Product of next layer's transposed weights and deltas
  double **help_2 = malloc((*layers) * sizeof(double *));
  // We do not need them at the output layer
  //////////////////////////////////////////////////////////////////
  if (deltas && grad_w && help_1 && help_2) {
    // Allocations
    for (l = (*layers) + 1; l--; ) {
      if (l == 0) {
        for_helper_batch = (*batch) * nodes[l];
        for_helper_w = (*cols_X) * nodes[l];
        help_1[l] = malloc((*batch) * nodes[l] * sizeof(double));
        help_2[l] = malloc((*batch) * nodes[l] * sizeof(double));
        memset(help_1[l], 0.0, (*batch) * nodes[l] * sizeof(double));
        memset(help_2[l], 0.0, (*batch) * nodes[l] * sizeof(double));
      } else if (l == (*layers)) {
        for_helper_batch = (*batch) * (*cols_Y);
        for_helper_w = nodes[l-1] * (*cols_Y);
      } else {
        for_helper_batch = (*batch) * nodes[l];
        for_helper_w = nodes[l-1] * nodes[l];
        help_1[l] = malloc((*batch) * nodes[l] * sizeof(double));
        help_2[l] = malloc((*batch) * nodes[l] * sizeof(double));
        memset(help_1[l], 0.0, (*batch) * nodes[l] * sizeof(double));
        memset(help_2[l], 0.0, (*batch) * nodes[l] * sizeof(double));
      }
      deltas[l] = malloc(for_helper_batch * sizeof(double));    
      grad_w[l] = malloc(for_helper_w * sizeof(double));
      if (deltas[l] && grad_w[l]) {
        memset(deltas[l], 0.0, for_helper_batch * sizeof(double));
        memset(grad_w[l], 0.0, for_helper_w * sizeof(double));
      } else {
        printf(KRED "\nFailed to allocate deltas, gradients or helpers. Aborting...\n" RESET);
        free(deltas);
        free(grad_w);
        free(help_2);
        free(help_1);
        return;
      }
    }
  } else {
    printf(KRED "\nFailed to allocate deltas, gradients or helpers. Aborting...\n" RESET);
    return;
  }
  
  register double ***Z = cpu_feedforward_cache(rows, cols_Y, cols_X, layers, X, w, nodes, f);
  
  // Big switch in case we have pure batch (do not allocate mini-batches)
  switch ((*batch) == (*rows)) {
    // True
    case 1:
      correction = (*learning_rate) * 1.0/(double)(*rows);
      
      // Training
      do {
        // Find deltas
        cpu_gd_delta(deltas, help_1, help_2, rows, cols_Y, layers, Y, Z, w, nodes, f);
        
        // Now we update the weights and biases
        //         #pragma omp parallel for
        for (l = (*layers) + 1; l--; ) {
          switch (l == 0) {
            case 1:
              cpu_threaded_update(X, deltas[l], grad_w[l], w[l], cols_X, &nodes[l], rows, &correction);
              continue;
            default:
              break;
          } 
          switch (l > 0 && l < (*layers)) {
            case 1:
              cpu_threaded_update(Z[1][l-1], deltas[l], grad_w[l], w[l], &nodes[l-1], &nodes[l], rows, &correction);
              continue;
            default:
              break;
          }
          switch (l == (*layers)) {
            case 1:
              cpu_threaded_update(Z[1][l-1], deltas[l], grad_w[l], w[l], &nodes[l-1], cols_Y, rows, &correction);
              break;
            default:
              break;
          }
        }
        
        // Update Zs with the new wb's
        cpu_feedforward_update(rows, cols_Y, cols_X, layers, Z, X, w, nodes, f);
        
        switch (f[(*layers)]) {
          // If softmax then cross entropy
          case 4:
            loss = xentropy(rows, cols_Y, Y, Z[1][(*layers)]);
            break;
          default:
            loss = rmse(rows, cols_Y, Y, Z[1][(*layers)]);
            break;
        }
        
        if (loss != loss) {
          printf(KRED "\nWe got NaN values during training. Aborting...\n" RESET);
          for (l = 0; l < (*layers) + 1; l++) {
            free(deltas[l]);
            free(grad_w[l]);
            if (l < (*layers)) {
              free(help_2[l]);
              free(help_1[l]);
            }
          }
          free(deltas);
          free(grad_w);
          free(funcs);
          free(help_2);
          free(help_1);
          delete_Z(layers, Z);
          return;
        }
        printf("\nLoss = %.10lf at epoch = %d\n", loss, (*epochs) - e);
      } while (--e >= 0);
      break;
    default:
      correction = (*learning_rate) * 1.0/(double)(*batch);
      
      // They need to be allocated once
      double **X_batch = malloc(r_over_b * sizeof(double *));
      double **Y_batch = malloc(r_over_b * sizeof(double *));
      
      if (X_batch && Y_batch) {
        for (i = r_over_b; i--; ) {
          X_batch[i] = malloc((*batch) * (*cols_X) * sizeof(double));
          Y_batch[i] = malloc((*batch) * (*cols_Y) * sizeof(double));
          if (X_batch[i] && Y_batch[i]) {
            memset(X_batch[i], 0.0, (*batch) * (*cols_X) * sizeof(double));
            memset(Y_batch[i], 0.0, (*batch) * (*cols_Y) * sizeof(double));
          } else {
            printf(KRED "\nFailed to allocate X or Y batches. Aborting...\n" RESET);
            free(X_batch);
            free(Y_batch);
            return;
          }
        }
      } else {
        printf(KRED "\nFailed to allocate X or Y batches. Aborting...\n" RESET);
        return;
      }
      
      i = r_over_b - 1;
      // Fill X_batch, Y_batch
      do {
        memcpy(X_batch[i], X + i * (*batch) * (*cols_X), (*batch) * (*cols_X) * sizeof(double));
        memcpy(Y_batch[i], Y + i * (*batch) * (*cols_Y), (*batch) * (*cols_Y) * sizeof(double));
      } while (--i >= 0);
      
      // The perceptrons' outputs
      double ***Z_batch = malloc(2 * sizeof(double **));
      if (Z_batch) {
        // Unactivated
        Z_batch[0] = malloc(((*layers) + 1) * sizeof(double *));
        // Activated
        Z_batch[1] = malloc(((*layers) + 1) * sizeof(double *));
        if (Z_batch[0] && Z_batch[1]) {
          for (l = (*layers) + 1; l--; ) {
            switch (l == (*layers)) {
              // Last layer
              case 1:
                Z_batch[0][l] = malloc((*batch) * (*cols_Y) * sizeof(double));
                Z_batch[1][l] = malloc((*batch) * (*cols_Y) * sizeof(double));
                if (Z_batch[0][l] && Z_batch[1][l]) {
                  memset(Z_batch[0][l], 0.0, (*batch) * (*cols_Y) * sizeof(double));
                  memset(Z_batch[1][l], 0.0, (*batch) * (*cols_Y) * sizeof(double));
                } else {
                  printf(KRED "\nFailed to allocate Z batches. Aborting...\n" RESET);
                  free(Z_batch[0]);
                  free(Z_batch[1]);
                  return;
                }
                continue;
              default:
                Z_batch[0][l] = malloc((*batch) * nodes[l] * sizeof(double));
                Z_batch[1][l] = malloc((*batch) * nodes[l] * sizeof(double));
                if (Z_batch[0][l] && Z_batch[1][l]) {
                  memset(Z_batch[0][l], 0.0, (*batch) * nodes[l] * sizeof(double));
                  memset(Z_batch[1][l], 0.0, (*batch) * nodes[l] * sizeof(double));
                } else {
                  printf(KRED "\nFailed to allocate Z batches. Aborting...\n" RESET);
                  free(Z_batch[0]);
                  free(Z_batch[1]);
                  return;
                }
                continue;
            }
          }
        } else {
          printf(KRED "\nFailed to allocate Z batches. Aborting...\n" RESET);
          free(Z_batch);
          return;
        }
      } else {
        printf(KRED "\nFailed to allocate Z batches. Aborting...\n" RESET);
        return;
      }
      
      do {
        i = r_over_b - 1;
        do {
          printf("     \t#%d/%d\n", i + 1, r_over_b);
          
          // Fill Z_batch with new Zs
          for (l = (*layers); l >= 0; l--) {
            switch (l == (*layers)) {
              case 1:
                memcpy(Z_batch[0][l], Z[0][l] + i * (*batch) * (*cols_Y), (*batch) * (*cols_Y) * sizeof(double));
                memcpy(Z_batch[1][l], Z[1][l] + i * (*batch) * (*cols_Y), (*batch) * (*cols_Y) * sizeof(double));
                break;
              default:
                memcpy(Z_batch[0][l], Z[0][l] + i * (*batch) * nodes[l], (*batch) * nodes[l] * sizeof(double));
                memcpy(Z_batch[1][l], Z[1][l] + i * (*batch) * nodes[l], (*batch) * nodes[l] * sizeof(double));
                break;
            }
          }
          
          // Fill the deltas
          cpu_gd_delta(deltas, help_1, help_2, batch, cols_Y, layers, Y_batch[i], Z_batch, w, nodes, f);
          
          // Now we update the weights and biases
          //           #pragma omp parallel for
          for (l = (*layers) + 1; l--; ) {
            switch (l == 0) {
              case 1:
                cpu_threaded_update(X, deltas[l], grad_w[l], w[l], cols_X, &nodes[l], batch, &correction);
                continue;
              default:
                break;
            }
            switch (l > 0 && l < (*layers)) {
              case 1:
                cpu_threaded_update(Z[1][l-1], deltas[l], grad_w[l], w[l], &nodes[l-1], &nodes[l], batch, &correction);
                continue;
              default:
                break;
            }
            switch (l == (*layers)) {
              case 1:
                cpu_threaded_update(Z[1][l-1], deltas[l], grad_w[l], w[l], &nodes[l-1], cols_Y, batch, &correction);
                break;
              default:
                break;
            }
          }
          
          // Do an update of Z's with the new wb's
          cpu_feedforward_update(rows, cols_Y, cols_X, layers, Z, X, w, nodes, f);
        } while (--i >= 0);
        
        switch (f[(*layers)]) {
          // If softmax then cross entropy
          case 4:
            loss = xentropy(rows, cols_Y, Y, Z[1][(*layers)]);
            break;
          default:
            loss = rmse(rows, cols_Y, Y, Z[1][(*layers)]);
            break;
        }
        
        if (loss != loss) {
          printf(KRED "\nWe got NaN values during training. Aborting...\n" RESET);
          // Free before quitting
          for (l = 0; l < (*layers) + 1; l++) {
            free(Z_batch[0][l]);
            free(Z_batch[1][l]);
            free(deltas[l]);
            free(grad_w[l]);
            if (l < (*layers)) {
              free(help_2[l]);
              free(help_1[l]);
            }
          }
          free(Z_batch[0]);
          free(Z_batch[1]);
          free(Z_batch);
          free(help_2);
          free(help_1);
          free(deltas);
          free(grad_w);
          
          for (i = 0; i < r_over_b; i++) {
            free(X_batch[i]);
            free(Y_batch[i]);
          }
          free(Y_batch);
          free(X_batch);
          
          free(funcs);
          delete_Z(layers, Z);
          return;
        }
        printf("\nLoss = %.10lf at epoch = %d\n", loss, (*epochs) - e);
      } while (--e >= 0);
      
      // End of batching
      for (l = 0; l < (*layers) + 1; l++) {
        free(Z_batch[0][l]);
        free(Z_batch[1][l]);
      }
      free(Z_batch[0]);
      free(Z_batch[1]);
      free(Z_batch);
      
      for (i = 0; i < r_over_b; i++) {
        free(X_batch[i]);
        free(Y_batch[i]);
      }
      free(Y_batch);
      free(X_batch);
      break;
  }
  
  // Save weights and free memory
  save_w(w, layers, nodes, cols_Y, cols_X);
  
  // Free the rest
  for (l = 0; l < (*layers) + 1; l++) {
    free(deltas[l]);
    free(grad_w[l]);
    if (l < (*layers)) {
      free(help_2[l]);
      free(help_1[l]);
    }
  }
  free(deltas);
  free(grad_w);
  free(f);
  free(help_2);
  free(help_1);
  delete_Z(layers, Z);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// End of PUBLIC API ///////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Definitions of forward declarated functions
static inline void swap(double *restrict a, double *restrict b)
{
  int temp = *a;
  *a = *b;
  *b = temp;
}

static inline double mean(const int *restrict rows, const double *restrict col)
{
  double sum = 0.0;
  int i = (*rows) - 1;
  do {
    sum += col[i];
  } while (--i >= 0);
  sum /= (double)(*rows);
  return sum;
}

static inline double stdev(const int *restrict rows, const double *restrict col, const double *restrict mean)
{
  double sumsq = 0.0, subtr;
  int i = (*rows) - 1;
  do {
    subtr = col[i] - (*mean);
    sumsq += subtr * subtr;
  } while (--i >= 0);
  return sqrt(sumsq/(double)((*rows) - 1));
}

static inline int *name2int(const int *restrict layers, char funcs[(*layers) + 1][30])
{
  int *names2ints = malloc(((*layers) + 1) * sizeof(int));
  int l = (*layers);
  
  do {
    switch (strcmp(funcs[l], "relu")) {
      case 0:
        names2ints[l] = 0;
        continue;
      default:
        break;
    }
    switch (strcmp(funcs[l], "logistic")) {
      case 0:
        names2ints[l] = 1;
        continue;
      default:
        break;
    }
    switch (strcmp(funcs[l], "linear")) {
      case 0:
        names2ints[l] = 2;
        continue;
      default:
        break;
    }
    switch (strcmp(funcs[l], "tanh")) {
      case 0:
        names2ints[l] = 3;
        continue;
      default:
        break;
    }
    switch (strcmp(funcs[l], "softmax")) {
      case 0:
        names2ints[l] = 4;
        continue;
      default:
        break;
    }
    // Leaky relu
    switch (strcmp(funcs[l], "lrelu")) {
      case 0:
        names2ints[l] = 5;
        continue;
      default:
        break;
    }
    switch (strcmp(funcs[l], "softplus")) {
      case 0:
        names2ints[l] = 6;
        continue;
      default:
        break;
    }
    switch (strcmp(funcs[l], "softsign")) {
      case 0:
        names2ints[l] = 7;
        continue;
      default:
        break;
    }
    switch (strcmp(funcs[l], "arctan")) {
      case 0:
        names2ints[l] = 8;
        continue;
      default:
        break;
    }
    //Inverse square root with a = 1
    switch (strcmp(funcs[l], "isru")) {
      case 0:
        names2ints[l] = 9;
        continue;
      default:
        break;
    }
    //Inverse sqrt linear unit \w a=1
    switch (strcmp(funcs[l], "isrlu")) {
      case 0:
        names2ints[l] = 10;
        continue;
      default:
        break;
    }
    switch (strcmp(funcs[l], "bent")) {
      case 0:
        names2ints[l] = 11;
        continue;
      default:
        break;
    }
    switch (strcmp(funcs[l], "sinus")) {
      case 0:
        names2ints[l] = 12;
        continue;
      default:
        break;
    }
    switch (strcmp(funcs[l], "sinusc")) {
      case 0:
        names2ints[l] = 13;
        continue;
      default:
        // Gaussian if nothing else
        names2ints[l] = 14;
        break;
    }
  } while (--l >= 0);
  return names2ints;
}

static inline void activate(double *restrict Y, const double *restrict X,
                            const int *restrict r, const int *restrict c,
                            const int *restrict f)
{
  int i = (*r) * (*c) - 1;
  switch ((*f)) {
    // Relu
    case 0:
      do {
        switch (X[i] < 0.0) {
          case 1:
            Y[i] = 0.0;
            continue;
          default:
            Y[i] = X[i];
            continue;
        }
      } while (--i >= 0);
      return;
    // Logistic
    case 1:
      do {
        Y[i] = 1/(1 + exp(-X[i]));
      } while (--i >= 0);
      return;
      // Linear
    case 2:
      memcpy(Y, X, (i + 1) * sizeof(double));
      return;
      // Tanh
    case 3:
      do {
        Y[i] = tanh(X[i]);
      } while (--i >= 0);
      return;
      // Softmax
    case 4: {
      double *e = malloc((*c) * sizeof(double));
      double e_X;
      int row = (*r) - 1, col;
      do {
        e_X = 0.0;
        col = (*c) - 1;
        do {
          e[col] = exp(X[row * (*c) + col]);
          e_X += e[col];
        } while (--col >= 0);
        col = (*c) - 1;
        do {
          Y[row * (*c) + col] = e[col]/e_X;
        } while (--col >= 0);
      } while (--row >= 0);
      free(e);
      return;
    }
    // Lrelu
    case 5:
      do {
        switch (X[i] < 0.0) {
          case 1:
            Y[i] = 0.01 * X[i];
            continue;
          default:
            Y[i] = X[i];
            continue;
        }
      } while (--i >= 0);
      return;
    // Softplus
    case 6:
      do {
        Y[i] = log(1 + exp(X[i]));
      } while (--i >= 0);
      return;
      // Softsign
    case 7:
      do {
        Y[i] = X[i]/(1 + fabs(X[i]));
      } while (--i >= 0);
      return;
      // Arctan
    case 8:
      do {
        Y[i] = atan(X[i]);
      } while (--i >= 0);
      return;
      // Isru
    case 9:
      do {
        Y[i] = X[i]/sqrt(1 + X[i] * X[i]);
      } while (--i >= 0);
      return;
      // Isrlu
    case 10:
      do {
        switch (X[i] < 0.0) {
          case 1:
            Y[i] = X[i]/sqrt(1 + X[i] * X[i]);
            continue;
          default:
            Y[i] = X[i];
            continue;
        }
      } while (--i >= 0);
      return;
    // Bent
    case 11:
      do {
        Y[i] = (sqrt(X[i] * X[i] + 1.0) - 1.0)/2.0 + X[i];
      } while (--i >= 0);
      return;
      // Sinus
    case 12:
      do {
        Y[i] = sin(X[i]);
      } while (--i >= 0);
      return;
      // Sinusc
    case 13:
      do {
        switch (X[i] == 0.0) {
          case 1:
            Y[i] = 1.0;
            continue;
          default:
            Y[i] = sin(X[i])/X[i];
            continue;
        }
      } while (--i >= 0);
      return;
    // Gauss
    default:
      do {
        Y[i] = exp(-(X[i] * X[i]));
      } while (--i >= 0);
      return;
  }
}

static inline void gradient(double *restrict Y, const double *restrict X,
                            const int *restrict r, const int *restrict c,
                            const int *restrict f)
{
  int i = (*r) * (*c) - 1;
  switch ((*f)) {
    // Relu
    case 0:
      do {
        switch (X[i] < 0.0) {
          case 1:
            Y[i] = 0.0;
            continue;
          default:
            Y[i] = 1.0;
            continue;
        }
      } while (--i >= 0);
      return;
    // Logistic
    case 1: {
      double y;
      do {
        y = 1/(1 + exp(-X[i]));
        Y[i] = y * (1 - y);
      } while (--i >= 0);
      return;
    }
    // Linear
    case 2:
      memset(Y, 1.0, (i + 1) * sizeof(double));
      return;
      // Tanh
    case 3: {
      double e_X, e_mX, y;
      do {
        e_X = exp(X[i]);
        e_mX = exp(-X[i]);
        y = (e_X - e_mX)/(e_X + e_mX);
        Y[i] = 1 - y * y;
      } while (--i >= 0);
      return;
    }
    // Softmax
    case 4: {
      double *e = malloc((*c) * sizeof(double));
      int row = (*r) - 1, col;
      double e_X, e_mX;
      do {
        e_X = 0.0;
        col = (*c) - 1;
        do {
          e[col] = exp(X[row * (*c) + col]);
          e_X += e[col];
        } while (--col >= 0);
        col = (*c) - 1;
        do {
          e_mX = e[col]/e_X;
          switch (row == col) {
            case 1:
              Y[row * (*c) + col] = e_mX * (1 - e_mX);
              break;
            default:
              Y[row * (*c) + col] = - e_mX * e_mX;
              break;
          }
        } while (--col >= 0);
      } while (--row >= 0);
      free(e);
      return;
    }
    // Lrelu
    case 5:
      do {
        switch (X[i] < 0.0) {
          case 1:
            Y[i] = 0.01;
            continue;
          default:
            Y[i] = 1.0;
            continue;
        }
      } while (--i >= 0);
      return;
    // Softplus
    case 6:
      do {
        Y[i] = 1/(1 + exp(-X[i]));
      } while (--i >= 0);
      return;
      // Softsign
    case 7: {
      double y;
      do {
        y = 1 + fabs(X[i]);
        Y[i] = 1/(y * y);
      } while (--i >= 0);
      return;
    }
    // Arctan
    case 8:
      do {
        Y[i] = 1/(X[i] * X[i] + 1);
      } while (--i >= 0);
      return;
      // Isru
    case 9: {
      double sq, y;
      do {
        sq = sqrt(1 + X[i] * X[i]);
        y = X[i]/sq;
        Y[i] = y * y * y;
      } while (--i >= 0);
      return;
    }
    // Isrlu
    case 10: {
      double sq, y;
      do {
        switch (X[i] < 0.0) {
          case 1:
            sq = sqrt(1 + X[i] * X[i]);
            y = X[i]/sq;
            Y[i] = y * y * y;
            continue;
          default:
            Y[i] = 1.0;
            continue;
        }
      } while (--i >= 0);
      return;
    }
    // Bent
    case 11: {
      double y, add;
      do {
        add = X[i] + 1;
        y = sqrt(add * add);
        Y[i] = X[i]/(2 * y) + 1;
      } while (--i >= 0);
      return;
    }
    // Sinus
    case 12:
      do {
        Y[i] = cos(X[i]);
      } while (--i >= 0);
      return;
      // Sinusc
    case 13:
      do {
        switch (X[i] == 0.0) {
          case 1:
            Y[i] = 0.0;
            continue;
          default:
            Y[i] = cos(X[i])/X[i] - sin(X[i])/(X[i] * X[i]);
            continue;
        }
      } while (--i >= 0);
      return;
    // Gauss
    default:
      do {
        Y[i] = -2.0 * X[i] * exp(-(X[i] * X[i]));
      } while (--i >= 0);
      return;
  }
}

static inline double rmse(const int *restrict r, const int *restrict c,
                          const double *restrict Y, const double *restrict Z)
{
  int i = (*r) * (*c) - 1;
  double loss = 0.0;
  double d;
  do {
    d = Z[i] - Y[i];
    loss += (d * d)/(double)(*c);
  } while (--i >= 0);
  loss = loss/(double)(*r);
  return sqrt(loss);
}

static inline double xentropy(const int *restrict r, const int *restrict c,
                              const double *restrict Y, const double *restrict Z)
{
  int i = (*r) * (*c) - 1;
  double loss = 0.0;
  do {
    loss += Y[i] * log(Z[i])/(double)(*c);
  } while (--i >= 0);
  return -loss/(double)(*r);
}

static inline void save_w(double **restrict w, const int *restrict layers, const int *restrict nodes,
                          const int *restrict cols_Y, const int *restrict cols_X)
{
  int l;
  char path[1024];
  char w_path[1036];
  getcwd(path, sizeof(path));
  strcpy(w_path, path);
  
  struct stat st = {0};
  if (stat(strcat(w_path, "/weights"), &st) == -1) {
    mkdir(w_path, 0700);
    printf("\nCreated ./wb/weights directory\n");
  }
  
  FILE *ptr_fp;
  
  for (l = 0; l < (*layers) + 1; l++) {
    
    // This needs to change every time
    char w_path_filename[1050];
    char number[100];
    char filename[15] = "layer_";
    sprintf(number, "%d", l);
    strcat(filename, number);
    strcat(filename, ".bin");
    
    strcpy(w_path_filename, w_path);
    strcat(w_path_filename, "/");
    
    strcat(w_path_filename, filename);
    
    if((ptr_fp = fopen(w_path_filename, "wb")) == NULL) {
      printf("Unable to create file!\n");
      exit(1);
    }
    
    if (l == 0) {
      if(fwrite(w[l], (*cols_X) * nodes[l] * sizeof(double), 1, ptr_fp) != 1) {
        printf("Write error!\n");
        exit(1);
      }
      fclose(ptr_fp);
    } else if (l == (*layers)) {
      if(fwrite(w[l], nodes[l-1] * (*cols_Y) * sizeof(double), 1, ptr_fp) != 1) {
        printf("Write error!\n");
        exit(1);
      }
      fclose(ptr_fp);
    } else {
      if(fwrite(w[l], nodes[l-1] * nodes[l] * sizeof(double), 1, ptr_fp) != 1) {
        printf("Write error!\n");
        exit(1);
      }
      fclose(ptr_fp);
    }
  }
}

static inline int ***imgpad(int ***restrict images, const int *restrict no_of_images,
                            const int *restrict img_width,
                            const int *restrict img_height,
                            const int *restrict img_channels,
                            const int *restrict padding, // Zeros around
                            const int *restrict delete_originals) // 0 = no, 1 = yes (keep only vector in memory)
{
  if ((*padding) == 0) {
    return images;
  } else {
    int image, i, j, rgb;
    int multiplication = ((*img_width) + 2 * (*padding)) * ((*img_height) + 2 * (*padding));
    
    int ***images_padded = malloc((*no_of_images) * sizeof(int **));
    
    for (image = (*no_of_images); image--; ) {
      images_padded[image] = malloc(multiplication * sizeof(int *));
      for (i = multiplication; i--; ) {
        images_padded[image][i] = malloc((*img_channels) * sizeof(int));
        for (rgb = (*img_channels); rgb--; ) {
          images_padded[image][i][rgb] = 0;
        }
      }
      for (i = (*img_height); i--; ) {
        for (j = (*img_width); j--; ) {
          for (rgb = (*img_channels); rgb--; ) {
            images_padded[image][(i + (*padding)) * ((*img_width) + 2 * (*padding)) + j + (*padding)][rgb] =
                                                                              images[image][i * (*img_width) + j][rgb];
          }
        }
      }
    }
    
    if ((*delete_originals) == 1) {
      multiplication = (*img_width) * (*img_height);
      for (i = 0; i < (*no_of_images); i++) {
        for (j = 0; j < multiplication; j++) {
          free(images[i][j]);
        }
        free(images[i]);
      }
      free(images);
    }
    return images_padded;
  }
}

static inline int **im2col(int ***restrict images,const int *restrict no_of_images,
                           const int *restrict img_width,
                           const int *restrict img_height,
                           const int *restrict img_channels,
                           const int *restrict spatial, // width and height of weights
                           const int *restrict stride, // (img_width - spatial + 2 * padding)/stride should be int
                           const int *restrict padding, // Zeros around
                           const int *restrict delete_originals) // 0 = no, 1 = yes (keep only vectorized in memory)
{
  int image, i, i_prime, j, pixel_x, pixel_y, multiplication;
  size_t rgb;
  // How many boxes horizontally
  int locations_width = ((*img_width) - (*spatial) + 2 * (*padding))/(*stride) + 1;
  // How many boxes vertically
  int locations_height = ((*img_height) - (*spatial) + 2 * (*padding))/(*stride) + 1;
  // Rows of vectorized image
  int receptive_fields = locations_width * locations_height;
  multiplication = (*spatial) * (*spatial) * (*img_channels);
  int ***images_w_pad = imgpad(images, no_of_images, img_width, img_height, img_channels, padding, delete_originals);
  int **images_as_matrices = malloc((*no_of_images) * sizeof(int *));
  for (image = (*no_of_images); image--; ) {
    pixel_x = 0;
    pixel_y = 0;
    i_prime = 0;
    // dim(receptive_fields X (spatial * spatial * img_channels))
    images_as_matrices[image] = malloc(receptive_fields * multiplication * sizeof(int));
    for (i = 0; i < receptive_fields; i++) {
      rgb = 0;
      if (i % locations_height == 0 && i > 0) {
        i_prime += 1;
        pixel_x = 0;
      } else if (i % locations_height != 0 && i > 0){
        pixel_x += (*stride);
      } else {
        pixel_x = 0;
      }
      pixel_y = i_prime * (*stride);
      for (j = 0; j < multiplication; j++) {
        // The channel change happens in every row for every conv box
        if (j > 0 && j % ((*spatial) * (*spatial)) == 0) {
          rgb += 1;
          if (rgb == (*img_channels)) {
            rgb = 0;
          }
        }
        // Width and height pixels
        if (j % (*spatial) == 0 && j > 0) {
          pixel_y += 1;
          if (j % ((*spatial) * (*spatial)) == 0) {
            pixel_y = i_prime * (*stride);
          }
          pixel_x -= (*spatial) - 1;
          if (i % locations_height == 0 && i > 0) {
            pixel_x = 0;
          }
        } else if (j % (*spatial) != 0 && j > 0) {
          pixel_x += 1;
        }
        /////////////////////////////////////////////////////////////
        // The operation
        images_as_matrices[image][i * multiplication + j] =
        images_w_pad[image][pixel_x * ((*img_width) + 2 * (*padding)) + pixel_y][rgb];
        if (j == multiplication - 1) {
          pixel_x -= (*spatial) - 1;
        }
      }
    }
  }
  
  if ((*padding) != 0) {
    multiplication = ((*img_width) + 2 * (*padding)) * ((*img_height) + 2 * (*padding));
    for (i = 0; i < (*no_of_images); i++) {
      for (j = 0; j < multiplication; j++) {
        free(images_w_pad[i][j]);
      }
      free(images_w_pad[i]);
    }
    free(images_w_pad);
  }
  
  if ((*delete_originals) == 1 && (*padding) == 0) {
    for (i = 0; i < (int)(*no_of_images); i++) {
      for (j = 0; j < (int)((*img_width) * (*img_height)); j++) {
        free(images[i][j]);
      }
      free(images[i]);
    }
    free(images);
  }
  return images_as_matrices;
}


static inline void delete_Z(const int *restrict layers, double ***restrict Z)
{
  int l;
  for (l = 0; l < (*layers) + 1; l++) {
    free(Z[1][l]);
    free(Z[0][l]);
  }
  free(Z[1]);
  free(Z[0]);
  free(Z);
}

static inline void delete_img_vector(const int *restrict no_of_images, int **restrict images)
{
  int image;
  for (image = 0; image < (*no_of_images); image++) free(images[image]);
  free(images);
}

static inline void cpu_feedforward_update(const int *restrict r, const int *restrict cY,
                                          const int *restrict cX, const int *restrict layers,
                                          double ***restrict Z,
                                          const double *restrict X, double **restrict w,
                                          const int *restrict n, const int *restrict f)
{
  // l is for layers
  int l = 0;
  do {
    switch (l == 0) {
      case 1:
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (*r), // Rows of z[0][l][0][j]
                    n[l], // Columns of z[0][l][0][j]
                    (*cX), // columns of A, rows of B
                    1.0, // scaling factor (none)
                    X, (*cX), // C = A * B -> matrix A, ldA
                    w[l], n[l], // C = A * B -> matrix B, ldB
                    0.0, // scaling factor for C (none)
                    Z[0][l], n[l]); // C, ldC
        activate(Z[1][l], Z[0][l], r, &n[l], &f[l]);
        ++l;
        continue;
      default:
        break;
    }
    switch (l > 0 && l < (*layers)) {
      case 1:
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (*r), // Rows of z[0][l][i][j]
                    n[l], // Columns of z[0][l][i][j]
                    n[l-1], // columns of A, rows of B
                    1.0, // scaling factor (none)
                    Z[1][l-1], n[l-1], // C = A * B -> matrix A, ldA
                    w[l], n[l], // C = A * B -> matrix B, ldB
                    0.0, // scaling factor for C (none)
                    Z[0][l], n[l]); // C, ldC
        activate(Z[1][l], Z[0][l], r, &n[l], &f[l]);
        ++l;
        continue;
      default:
        break;
    }
    switch (l == (*layers)) {
      case 1:
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (*r), // Rows of z[0][l][i][j]
                    (*cY), // Columns of z[0][l][0][j]
                    n[l-1], // columns of A, rows of B
                    1.0, // scaling factor (none)
                    Z[1][l-1], n[l-1], // C = A * B -> matrix A, ldA
                    w[l], (*cY), // C = A * B -> matrix B, ldB
                    0.0, // scaling factor for C (none)
                    Z[0][l], (*cY)); // C, ldC
        activate(Z[1][l], Z[0][l], r, cY, &f[l]);
        return;
    }
  } while (l <= (*layers));
}

static inline double ***cpu_feedforward_cache(const int *restrict r, const int *restrict cY,
                                              const int *restrict cX, const int *restrict layers,
                                              const double *restrict X, double **restrict w,
                                              const int *restrict n, const int *restrict f)
{
  // l is for layers
  // i is for each row * column of X, Y
  int l = (*layers);
  int i = (*r) * (*cY);
  // feeds at every layer
  double ***Z = malloc(2 * sizeof(double **));
  if (Z) {
    Z[0] = malloc(((*layers) + 1) * sizeof(double *));
    Z[1] = malloc(((*layers) + 1) * sizeof(double *));
    if (Z[0] && Z[1]) {
      Z[0][l] = malloc(i * sizeof(double));
      Z[1][l] = malloc(i * sizeof(double));
      if (Z[0][l] && Z[1][l]) {
        for (l = 0; l < (*layers); l++) {
          i = (*r) * n[l];
          Z[0][l] = malloc(i * sizeof(double));
          Z[1][l] = malloc(i * sizeof(double));
          if (Z[0][l] && Z[1][l]) {
            memset(Z[0][l], 0.0, i * sizeof(double));
            memset(Z[1][l], 0.0, i * sizeof(double));
          } else {
            printf(KRED "\nCould not allocate Zs. Aborting...\n" RESET);
            free(Z[0]);
            free(Z[1]);
            free(Z);
            abort();
          }
        }
      } else {
        printf(KRED "\nCould not allocate Zs. Aborting...\n" RESET);
        free(Z[0]);
        free(Z[1]);
        free(Z);
        abort();
      }
    } else {
      printf(KRED "\nCould not allocate Zs. Aborting...\n" RESET);
      free(Z);
      abort();
    }
  } else {
    printf(KRED "\nCould not allocate Zs. Aborting...\n" RESET);
    abort();
  }
  // Directly manipulates Z
  cpu_feedforward_update(r, cY, cX, layers, Z, X, w, n, f);
  return Z;
}

static inline void cpu_gd_delta(double **restrict d, double **restrict h1, double **restrict h2,
                                const int *restrict r, const int *restrict c, const int *restrict layers,
                                const double *restrict Y, double ***restrict Z, double **restrict w,
                                const int *restrict n, const int *restrict f)
{
  int l, i = (*r) * (*c) - 1;
  // Last layer
  switch (f[(*layers)]) {
    // Linear case
    case 2:
      do {
        d[(*layers)][i] = Z[1][(*layers)][i] - Y[i];
      } while (--i >= 0);
      break;
      // Softmax Crossentropy
    case 4:
      do {
        d[(*layers)][i] = Z[1][(*layers)][i] - Y[i];
      } while (--i >= 0);
      break;
    default:
      gradient(d[(*layers)], Z[0][(*layers)], r, c, &f[(*layers)]);
      do {
        d[(*layers)][i] *= Z[1][(*layers)][i] - Y[i];
      } while (--i >= 0);
      break;
  }
  // Before last
  l = (*layers) - 1;
  i = (*r) * n[l] - 1;
  gradient(h1[l], Z[0][l], r, &n[l], &f[l]);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              (*r), // Rows of help_2[0][j]
              n[l], // Columns of help_2[0][j]
              (*c), // columns of A, rows of B
              1.0, // scaling factor (none)
              d[l+1], (*c), // C = A * B -> matrix A, ldA
              w[l+1], (*c), // C = A * B -> matrix B, ldB
              0.0, // scaling factor for C (none)
              h2[l], n[l]); // C, ldC
  // Hadamard product
  do {
    d[l][i] = h1[l][i] * h2[l][i];
  } while (--i >= 0);
  switch ((*layers) == 1) {
    case 1:
      return;
    default:
      // All other layers
      l = (*layers) - 2;
      do {
        i = (*r) * n[l] - 1;
        gradient(h1[l], Z[0][l], r, &n[l], &f[l]);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (*r), // Rows of help_2[0][j]
                    n[l], // Columns of help_2[0][j]
                    n[l+1], // columns of A, rows of B
                    1.0, // scaling factor (none)
                    d[l+1], n[l+1], // C = A * B -> matrix A, ldA
                    w[l+1], n[l+1], // C = A * B -> matrix B,  ldB
                    0.0, // scaling factor for C (none)
                    h2[l], n[l]); // C, ldC
        // Hadamard product
        do {
          d[l][i] = h1[l][i] * h2[l][i];
        } while (--i >= 0);
      } while (--l >= 0);
      return;
  }
}

// returns new wb
static inline void cpu_threaded_update(const double *restrict X, const double *restrict d,
                                       double *restrict gw,
                                       double *restrict w,
                                       const int *restrict m,
                                       const int *restrict n,
                                       const int *restrict k,
                                       const double *restrict c)
{  
  int i = (*m) * (*n) - 1;
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
              (*m), // Rows of grad_w[l]
              (*n), // Columns of grad_w[l]
              (*k), // columns of A, rows of B
              (*c), // scaling factor
              X, (*m), // C = A * B -> matrix A, ldA
              d, (*n), // C = A * B -> matrix B, ldB
              0.0, // scaling factor for C (none)
              gw, (*n)); // C, ldC
  do {
    w[i] -= gw[i];
  } while (--i >= 0);
}


#endif  // libartificial_h__
