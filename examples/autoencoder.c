#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "../src/headers/neurons.h"
#include "../src/headers/utils.h"
#include "../src/headers/training.h"

#include "../src/headers/prints.h"

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
  normalize(X, rows, columns_X);
  
  // Then we randomize the inputs
  randomize(X, rows, columns_X);
  
  // We initialize weights and biases at every layer (if we do not already have them)
  // wb[0] the weights
  // wb[1] the biases
  // wb[0][l][i * columns_X + j] weights at layer l=0,...,layers, i'th row j'th column
  // wb[1][l][j] biases at layer l=0,...,layers always 1 row and j'th column
  double ***wb = init_wb(w_variance, layers, nodes, funcs, columns_Y, columns_X);
  
  // 	showWB(layers, nodes, columns_Y, columns_X, wb);
  
  // The outputs from neurons
  // We care about Z[1][layers][i * columns + j] which is the final prediction
  // The rest are used for the updating
  double ***Z = feedforward(rows, columns_Y, columns_X, layers, X, wb, nodes, funcs);
  
  // All the updating in one function (manipulates wb)
  update_gd(rows, columns_Y, columns_X, batch, layers, nodes, X, X, Z, wb, funcs, learning_rate, epochs);
  
  // 	showWB(layers, nodes, columns_Y, columns_X, wb);
  
  ////////////////////////////////////////////////////////////
  // Freeing stuff
  ////////////////////////////////////////////////////////////
  delete_Z(layers, Z);
  delete_wb(layers, wb);
  
  free(X);
  ////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////
  return 0;
}
