#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "../src/headers/utils.h"
#include "../src/headers/training_gpu.h"
#include "../src/headers/neurons_gpu.h"

int main(void)
{
  int i, j;
  
  // The model
  ///////////////////////////////////////////////////////////////////////////
  const size_t columns_X = 3;
  const size_t columns_Y = 1;
  const size_t rows = 1024;
  
  double *X = calloc(rows * columns_X, sizeof(double));
  
  srand(time(NULL));
  
  for(i = 0; i < (rows * columns_X); i++) {
    X[i] = rand_normal(30.0, 2.0);
  }
  
  double *Y = malloc(rows * columns_Y * sizeof(double));
  for (i = 0; i < rows; i++) {
    Y[i] = 8 * pow(X[i * columns_X], 2) - 5 * sqrt(X[i * columns_X + 1]) - sin(X[i * columns_X + 2]) + rand_normal(0,1);
  }
  ///////////////////////////////////////////////////////////////////////////
  
  // Hyperparameters
  ///////////////////////////////////////////////////////////////////////////
  const int batch = 256; // Divisor of 1024
  const double w_variance = 0.01; // For the weight initialization
  const double learning_rate = 0.0000000001;
  const int epochs = 5;
  
  const int layers = 7;
  const int nodes[7] = {3200, 2309, 397, 540, 408, 390, 480};
  char funcs[7 + 1][30] = {
    "logistic",
    "relu",
    "tanh",
    "lrelu",
    "gauss",
    "softmax",
    "gauss",
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
  
//   	int layers = 1;
//   	int nodes[1] = {200};
//   	char funcs[2][30] = {
//   		"gauss",
//   		"linear" // Regression and not classification (if classification something other than linear)
//   	};
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
  
  //   double ***wb = load_wb(layers, nodes, columns_Y, columns_X);
    
  // The outputs from neurons
  // We care about Z[1][layers][i * columns + j] which is the final prediction
  // The rest are used for the updating
  double ***Z = gpu_feedforward(rows, columns_Y, columns_X, layers, X, wb, nodes, funcs);
  /*
  printf("\n");
  for (i = 0; i < rows-1000; i++) {
    for (j = 0; j < columns_Y; j++) {
      printf("%f\t", Z[1][layers][i * columns_Y + j]);
    }
    printf("\n");
  }*/
  
  // All the updating in one function (manipulates wb)
  gpu_gd_update(rows, columns_Y, columns_X, batch, layers, nodes, Y, X, Z, wb, funcs, learning_rate, epochs);
  
  //   save_wb(wb, layers, nodes, columns_Y, columns_X);
  
  ////////////////////////////////////////////////////////////
  // Freeing stuff
  ////////////////////////////////////////////////////////////
  delete_Z(layers, Z);
  delete_wb(layers, wb);
  
  free(X);
  free(Y);
  ////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////
  return 0;
}
