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



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "../libartificial.h"

int main(void)
{
  int i, j;
  
  // The model
  ///////////////////////////////////////////////////////////////////////////
  const int columns_X = 3;
  const int columns_Y = 1;
  const int rows = 1024;
  
  double *X = malloc(rows * columns_X * sizeof(double));
  
  srand(time(NULL));
  
  // First column is a column of ones
  for (i = 0; i < rows; i++) {
    for (j = 0; j < columns_X; j++) {
      if (j == 0) {
        X[i * columns_X + j] = 1;
      } else if (j == 1) {
        X[i * columns_X + j] = rand_normal(20.0, 5.0);
      } else {
        X[i * columns_X + j] = rand_normal(3.0, 10.0);
      }
    }
  }
  
  double *Y = malloc(rows * columns_Y * sizeof(double));
  for (i = 0; i < rows; i++) {
    Y[i] = 8 * pow(X[i * columns_X], 2) + 5 * sqrt(X[i * columns_X + 1]) - sin(X[i * columns_X + 2]) + rand_normal(0,1);
  }
  ///////////////////////////////////////////////////////////////////////////
  
  // Hyperparameters
  ///////////////////////////////////////////////////////////////////////////
  const int batch = 256; // Divisor of 1024
  const double w_variance = 1; // For the weight initialization
  const double learning_rate = 0.000001;
  const int epochs = 500;
  
//   const int layers = 7;
//   const int nodes[7] = {308, 293, 392, 563, 445, 392, 481};
//   char funcs[7 + 1][30] = {
//     "logistic",
//     "tanh",
//     "gauss",
//     "bent",
//     "bent",
//     "softmax",
//     "gauss",
//     "linear" // Regression and not classification (if classification something other than linear)
//   };
//   
//   const int layers = 3;
//   const int nodes[3] = {6200, 99, 39};
//   char funcs[4][30] = {
//     "logistic",
//     "logistic",
//     "logistic",
//     "linear" // Regression and not classification (if classification something other than linear)
//   };
//   
//   const int layers = 2;
//   const int nodes[2] = {602, 39};
//   char funcs[3][30] = {
//     "softplus",
//     "gauss",
//     "linear" // Regression and not classification (if classification something other than linear)
//   };
  
  const int layers = 1;
  const int nodes[1] = {200};
  char funcs[2][30] = {
    "gauss",
    "linear" // Regression and not classification (if classification something other than linear)
  };
  
  // The procedure
  ////////////////////////////////////////////////////////////////////////////
  
  // First we normalize X for the gradients
  normalize(X, &rows, &columns_X);
  
  // Then we randomize the inputs
//   randomize(X, &rows, &columns_X);
    
  // We initialize weights and biases at every layer (if we do not already have them)
  // wb[l][i * columns_X + j] weights at layer l=0,...,layers, i'th row j'th column
  // wb[l][j] biases at layer l=0,...,layers always 1 row and j'th column
  double **weights = init_w(&w_variance, &layers, nodes, funcs, &columns_Y, &columns_X);
  
  // If you have already saved weights and biases
//   double **weights = load_w(&layers, &nodes, &columns_Y, &columns_X);
  
  // All the updating in one function (manipulates wb and saves it by default)
  cpu_gd_train(&rows, &columns_Y, &columns_X, &batch, &layers, nodes, Y, X, weights, funcs, &learning_rate, &epochs);
  
  double *Z = cpu_feedforward_predict(&rows, &columns_Y, &columns_X, &layers, X, weights, nodes, funcs);
  
  ////////////////////////////////////////////////////////////
  // Freeing stuff
  ////////////////////////////////////////////////////////////
  delete_w(&layers, weights);
  free(X);
  free(Y);
  free(Z);
  ////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////
  return 0;
}
