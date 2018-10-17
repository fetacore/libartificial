#include <string.h>
#include <stdlib.h>
#include <math.h>

void gradient(double *restrict X_graded, const double *restrict X,
              const int *restrict rows, const int *restrict cols, const int *restrict function) {
  
  int i = (*rows) * (*cols) - 1;
  
  switch ((*function)) {
    
    // Relu
    case 0:
      do {
        switch (X[i] < 0.0) {
          case 1:
            X_graded[i] = 0.0;
            continue;
          default:
            X_graded[i] = 1.0;
            continue;
        }
      } while (--i >= 0);
      return;
    
    // Logistic
    case 1: {
      double y;
      do {
        y = 1/(1 + exp(-X[i]));
        X_graded[i] = y * (1 - y);
      } while (--i >= 0);
      return;
    }
    
    // Linear
    case 2:
      memset(X_graded, 1.0, (i + 1) * sizeof(double));
      return;
    
    // Tanh
    case 3: {
      double e_X, e_mX, y;
      do {
        e_X = exp(X[i]);
        e_mX = exp(-X[i]);
        y = (e_X - e_mX)/(e_X + e_mX);
        X_graded[i] = 1 - y * y;
      } while (--i >= 0);
      return;
    }
    
    // Softmax
    case 4: {
      double *expos = malloc((*cols) * sizeof(double));
      int row = (*rows) - 1, col;
      double e_X, e_mX;
      do {
        e_X = 0.0;
        col = (*cols) - 1;
        do {
          expos[col] = exp(X[row * (*cols) + col]);
          e_X += expos[col];
        } while (--col >= 0);
        col = (*cols) - 1;
        do {
          e_mX = expos[col]/e_X;
          switch (row == col) {
            case 1:
              X_graded[row * (*cols) + col] = e_mX * (1 - e_mX);
              break;
            default:
              X_graded[row * (*cols) + col] = - e_mX * e_mX;
              break;
          }
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
            X_graded[i] = 0.01;
            continue;
          default:
            X_graded[i] = 1.0;
            continue;
        }
      } while (--i >= 0);
      return;
    
    // Softplus
    case 6:
      do {
        X_graded[i] = 1/(1 + exp(-X[i]));
      } while (--i >= 0);
      return;
    
    // Softsign
    case 7: {
      double y;
      do {
        y = 1 + fabs(X[i]);
        X_graded[i] = 1/(y * y);
      } while (--i >= 0);
      return;
    }
    
    // Arctan
    case 8:
      do {
        X_graded[i] = 1/(X[i] * X[i] + 1);
      } while (--i >= 0);
      return;
    
    // Isru
    case 9: {
      double sq, y;
      do {
        sq = sqrt(1 + X[i] * X[i]);
        y = X[i]/sq;
        X_graded[i] = y * y * y;
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
            X_graded[i] = y * y * y;
            continue;
          default:
            X_graded[i] = 1.0;
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
        X_graded[i] = X[i]/(2 * y) + 1;
      } while (--i >= 0);
      return;
    }
    
    // Sinus
    case 12:
      do {
        X_graded[i] = cos(X[i]);
      } while (--i >= 0);
      return;
    
    // Sinusc
    case 13:
      do {
        switch (X[i] == 0.0) {
          case 1:
            X_graded[i] = 0.0;
            continue;
          default:
            X_graded[i] = cos(X[i])/X[i] - sin(X[i])/(X[i] * X[i]);
            continue;
        }
      } while (--i >= 0);
      return;
    
    // Gauss
    default:
      do {
        X_graded[i] = -2.0 * X[i] * exp(-(X[i] * X[i]));
      } while (--i >= 0);
      return;
  }
}
