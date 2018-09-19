#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../headers/utils.h"

#define KGRN  "\x1B[32m"
#define RESET "\033[0m"

//	Variance is needed since depending on the data, tanh/relu may give nans.
//	Variance < 1 and close to 0.01 if data range too large

double ***init_wb(const double variance, const int layers,
                  const int nodes[layers], char funcs[layers+1][30],
                  const size_t columns_Y, const size_t columns_X)
{
  // l layers
  // i rows
  int l, i, multiplication;
  
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
            multiplication = (int)columns_X * nodes[l];
            wb[0][l] = malloc(multiplication * sizeof(double));
            wb[1][l] = malloc(nodes[l] * sizeof(double));
            
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
            
            for (i = 0; i < multiplication; i++) {
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
          multiplication = nodes[l-1] * columns_Y;
          wb[0][l] = malloc(multiplication * sizeof(double));
          wb[1][l] = malloc(columns_Y * sizeof(double));
          
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
          
          for (i = 0; i < multiplication; i++) {
            wb[0][l][i] = correction * rand_normal(0.0, variance);
            switch (i < (int)columns_Y) {
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
        multiplication = nodes[l-1] * nodes[l];
        wb[0][l] = malloc(multiplication * sizeof(double));
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
        
        for (i = 0; i < multiplication; i++) {
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
  printf(KGRN "\nWeights and biases initialized successfully!\n" RESET);
  return wb;
}
