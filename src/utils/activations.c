#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void activate(double *restrict X_activated, const double *restrict X,
              const int *restrict rows, const int *restrict cols, const int *restrict function) {
  
  int i = (*rows) * (*cols) - 1;
  
  switch ((*function)) {
    
    // Relu
    case 0:
      do {
        switch (X[i] < 0.0) {
          case 1:
            X_activated[i] = 0.0;
            continue;
          default:
            X_activated[i] = X[i];
            continue;
        }
      } while (--i >= 0);
      return;
    
    // Logistic
    case 1:
      do {
        X_activated[i] = 1/(1 + exp(-X[i]));
      } while (--i >= 0);
      return;
    
    // Linear
    case 2:
      memcpy(X_activated, X, (i + 1) * sizeof(double));
      return;
    
    // Tanh
    case 3:
      do {
        X_activated[i] = tanh(X[i]);
      } while (--i >= 0);
      return;
    
    // Softmax
    case 4: {
      double *expos = malloc((*cols) * sizeof(double));
      double e_X;
      int row = (*rows) - 1, col;
      do {
        e_X = 0.0;
        col = (*cols) - 1;
        do {
          expos[col] = exp(X[row * (*cols) + col]);
          e_X += expos[col];
        } while (--col >= 0);
        col = (*cols) - 1;
        do {
          X_activated[row * (*cols) + col] = expos[col]/e_X;
        } while (--col >= 0);
      } while (--row >= 0);
      free(expos);
      return;
    }
    
    // Lrelu
    case 5:
      do {
        switch (X[i] < 0.0) {
          case 1:
            X_activated[i] = 0.01 * X[i];
            continue;
          default:
            X_activated[i] = X[i];
            continue;
        }
      } while (--i >= 0);
      return;
    
    // Softplus
    case 6:
      do {
        X_activated[i] = log(1 + exp(X[i]));
      } while (--i >= 0);
      return;
    
    // Softsign
    case 7:
      do {
        X_activated[i] = X[i]/(1 + fabs(X[i]));
      } while (--i >= 0);
      return;
    
    // Arctan
    case 8:
      do {
        X_activated[i] = atan(X[i]);
      } while (--i >= 0);
      return;
    
    // Isru
    case 9:
      do {
        X_activated[i] = X[i]/sqrt(1 + X[i] * X[i]);
      } while (--i >= 0);
      return;
    
    // Isrlu
    case 10:
      do {
        switch (X[i] < 0.0) {
          case 1:
            X_activated[i] = X[i]/sqrt(1 + X[i] * X[i]);
            continue;
          default:
            X_activated[i] = X[i];
            continue;
        }
      } while (--i >= 0);
      return;
    
    // Bent
    case 11:
      do {
        X_activated[i] = (sqrt(X[i] * X[i] + 1.0) - 1.0)/2.0 + X[i];
      } while (--i >= 0);
      return;
    
    // Sinus
    case 12:
      do {
        X_activated[i] = sin(X[i]);
      } while (--i >= 0);
      return;
    
    // Sinusc
    case 13:
      do {
        switch (X[i] == 0.0) {
          case 1:
            X_activated[i] = 1.0;
            continue;
          default:
            X_activated[i] = sin(X[i])/X[i];
            continue;
        }
      } while (--i >= 0);
      return;
    
    // Gauss
    default:
      do {
        X_activated[i] = exp(-(X[i] * X[i]));
      } while (--i >= 0);
      return;
  }
}
