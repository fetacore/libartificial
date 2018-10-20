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
  int i;
  
  // The model
  ///////////////////////////////////////////////////////////////////////////
  int columns_X = 3;
  int columns_Y = 3;
  int rows = 1024;
  
  double *X = calloc(rows * columns_X, sizeof(double));
  
  srand(time(NULL));
  
  for(i = 0; i < (rows * columns_X); i++) {
    X[i] = rand_normal(30.0, 2.0);
  }
  ///////////////////////////////////////////////////////////////////////////
  
  // Hyperparameters
  int batch = 512; // Divisor of 1024
  double w_variance = 0.1; // For the weight initialization
  double learning_rate = 0.000000001;
  int epochs = 100;
  
  int layers = 7;
  int nodes[7] = {62, 230, 397, 540, 408, 390, 408};
  char funcs[8][30] = {
    "logistic",
    "relu",
    "tanh",
    "lrelu",
    "gauss",
    "softmax",
    "softmax",
    "linear" // Regression and not classification (if classification something other than linear)
  };
  
  // 	int layers = 3;
  // 	int nodes[3] = {6200, 9900, 3900};
  // 	char funcs[4][30] = {
  // 		"logistic",
  // 		"logistic",
  // 		"logistic",
  // 		"linear" // Regression and not classification (if classification something other than linear)
  // 	};
  
  // 	int layers = 2;
  // 	int nodes[2] = {602, 399};
  // 	char funcs[3][30] = {
  // 		"logistic",
  // 		"gauss",
  // 		"linear" // Regression and not classification (if classification something other than linear)
  // 	};
  
  // 	int layers = 1;
  // 	int nodes[1] = {200};
  // 	char funcs[2][30] = {
  // 		"gauss",
  // 		"linear" // Regression and not classification (if classification something other than linear)
  // 	};
  ////////////////////////////////////////////////////////////////////////////
  
  // The procedure
  ////////////////////////////////////////////////////////////////////////////
  
  // First we normalize X for the gradients
  normalize(X, &rows, &columns_X);
  
  // Then we randomize the inputs
  randomize(X, &rows, &columns_X);
  
  // We initialize weights and biases at every layer (if we do not already have them)
  // wb[l][i * columns_X + j] weights at layer l=0,...,layers, i'th row j'th column
  // wb[l][j] biases at layer l=0,...,layers always 1 row and j'th column
  double **weights = init_w(&w_variance, &layers, nodes, funcs, &columns_Y, &columns_X);
    
  // If you have already saved weights and biases
  //   double **weights = load_w(&layers, &nodes, &columns_Y, &columns_X);
  
  // All the updating in one function (manipulates wb and saves it by default)
  cpu_gd_train(&rows, &columns_Y, &columns_X, &batch, &layers, nodes, X, X, weights, funcs, &learning_rate, &epochs);
  
  double *Z = cpu_feedforward_predict(&rows, &columns_Y, &columns_X, &layers, X, weights, nodes, funcs);
  
  ////////////////////////////////////////////////////////////
  // Freeing stuff
  ////////////////////////////////////////////////////////////
  delete_w(&layers, weights);
  free(X);
  free(Z);
  ////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////
  return 0;
}
