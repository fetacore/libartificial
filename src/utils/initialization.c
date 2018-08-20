#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../headers/utils.h"

//	Variance is needed since depending on the data, tanh/relu may give nans.
//	Variance < 1 and close to 0.01 if data range too large

double ***init_wb(double variance, int layers, int nodes[layers], char funcs[layers][30], int columns_Y, int columns_X)
{
  // l layers
  // i rows
  int l, i;
  
  // wb[0] is weights;
  // wb[1] is biases;
  
  double ***wb = malloc(2 * sizeof(double **));
  wb[0] = malloc((layers + 1) * sizeof(double *));
  wb[1] = malloc((layers + 1) * sizeof(double *));
  
  // For the heuristics of weight initialization
  double correction;
  
  for (l = 0; l < layers + 1; l++) {
    int isRelu = strcmp(funcs[l], "relu") + strcmp(funcs[l], "lrelu");
    int isTanh = strcmp(funcs[l], "tanh");
    
    switch (l > 0 && l < layers) {
      // The statement is false
      case 0:
        switch (l == 0) {
          // The statement is true
          case 1:
            wb[1][l] = malloc(nodes[l] * sizeof(double));
            wb[0][l] = malloc(columns_X * nodes[l] * sizeof(double));
            
            switch (isRelu == 2 && isTanh == 1) {
              // One of the two is false
              case 0:
                switch (isRelu == 1) {
                  // Either relu or lrelu
                  case 1:
                    // He et al.
                    correction = sqrt(2.0/(double)columns_X);
                    break;
                  default:
                    // Xavier
                    correction = sqrt(1.0/(double)columns_X);
                    break;
                }
                break;
              default:
                correction = sqrt(2.0/(double)(columns_X + nodes[l]));
                break;
            }
            
            for (i = 0; i < columns_X * nodes[l]; i++) {
              wb[0][l][i] = correction * rand_normal(0.0, variance);
              switch (i < nodes[l]) {
                case 1:
                  wb[1][l][i] = 0.0;
                  break;
                default:
                  break;
              }
            }
            break;
            
        // l = layers
        default:
          wb[1][l] = malloc(columns_Y * sizeof(double));
          wb[0][l] = malloc(nodes[l-1] * columns_Y * sizeof(double));
          
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
              correction = sqrt(2.0/(double)(nodes[l-1] + columns_Y));
              break;
          }
          
          for (i = 0; i < nodes[l-1] * columns_Y; i++) {
            wb[0][l][i] = correction * rand_normal(0.0, variance);
            switch (i < columns_Y) {
              case 1:
                wb[1][l][i] = 0.0;
                break;
              default:
                break;
            }
          }
          break;
        }
        break;
        
      // We are in between input and output
      default:
        
        wb[0][l] = malloc(nodes[l-1] * nodes[l] * sizeof(double));
        wb[1][l] = malloc(nodes[l] * sizeof(double));
        
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
            correction = sqrt(2.0/(double)(nodes[l-1] + nodes[l]));
            break;
        }
        
        for (i = 0; i < nodes[l-1] * nodes[l]; i++) {
          wb[0][l][i] = correction * rand_normal(0.0, variance);
          switch (i < nodes[l]) {
            case 1:
              wb[1][l][i] = 0.0;
              break;
            default:
              break;
          }
        }
        break;
    }
  }
  return wb;
}
