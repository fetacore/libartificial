#include <string.h>
#include <math.h>

int i;

void gradient(double *X_graded, double *X, int threshold, char *function) {
  
  switch (strcmp(function, "relu")) {
    case 0:
      for (i = threshold; i--; ) {
        switch (X[i] < 0.0) {
          case 1:
            X_graded[i] = 0.0;
            break;
          default:
            X_graded[i] = 1.0;
            break;
        }
      }
      return;
    default:
      break;
  }
  
  // Leaky relu
  switch (strcmp(function, "lrelu")) {
    case 0:
      for (i = threshold; i--; ) {
        switch (X[i] < 0.0) {
          case 1:
            X_graded[i] = 0.01;
            break;
          default:
            X_graded[i] = 1.0;
            break;
        }
      }
      return;
    default:
      break;
  }
  
  double e_X = 0, e_mX = 0, y = 0;
  switch (strcmp(function, "tanh")) {
    case 0:
      for (i = threshold; i--; ) {
        e_X = exp(X[i]);
        e_mX = exp(-X[i]);
        y = (e_X - e_mX)/(e_X + e_mX);
        X_graded[i] = 1 - y * y;
      }
      return;
    default:
      break;
  }
  
  switch (strcmp(function, "logistic")) {
    case 0:
      for (i = threshold; i--; ) {
        e_mX = exp(-X[i]);
        y = 1/(1 + e_mX);
        X_graded[i] = y * (1 - y);
      }
      return;
    default:
      break;
  }
  
  switch (strcmp(function, "linear")) {
    case 0:
      for (i = threshold; i--; ) {
        X_graded[i] = 1.0;
      }
      return;
    default:
      break;
  }
  
  switch (strcmp(function, "softmax")) {
    case 0:
      for (i = threshold; i--; ) {
        // Abuse of notation (sum of all exp(X_i))
        e_X += exp(X[i]);
      } 
      for (i = threshold; i--; ) {
        // Abuse again
        e_mX = exp(X[i])/e_X;
        X_graded[i] = e_mX * (1 - e_mX);
      }
      return;
    default:
      break;
  }
  
  switch (strcmp(function, "softplus")) {
    case 0:
      for (i = threshold; i--; ) {
        e_mX = exp(-X[i]);
        X_graded[i] = 1/(1 + e_mX);
      }
      return;
    default:
      break;
  }
  
  switch (strcmp(function, "softsign")) {
    case 0:
      for (i = threshold; i--; ) {
        // abuse of notation
        e_X = fabs(X[i]);
        y = 1 + e_X;
        X_graded[i] = 1/(y * y);
      }
      return;
    default:
      break;
  }
  
  switch (strcmp(function, "arctan")) {
    case 0:
      for (i = threshold; i--; ) {
        X_graded[i] = 1/(X[i] * X[i] + 1);
      }
      return;
    default:
      break;
  }
  
  //Inverse square root with a = 1
  switch (strcmp(function, "isru")) {
    case 0:
      for (i = threshold; i--; ) {
        y = sqrt(1 + X[i] * X[i]);
        y = X[i]/y;
        X_graded[i] = y * y * y;
      }
      return;
    default:
      break;
  }
  
  //Inverse sqrt linear unit \w a=1
  switch (strcmp(function, "isrlu")) {
    case 0:
      for (i = threshold; i--; ) {
        switch (X[i] < 0.0) {
          case 1:
            y = sqrt(1 + X[i] * X[i]);
            y = X[i]/y;
            X_graded[i] = y * y * y;
            break;
          default:
            X_graded[i] = 1.0;
            break;
        }
      }
      return;
    default:
      break;
  }
  
  switch (strcmp(function, "bent")) {
    case 0:
      for (i = threshold; i--; ) {
        // abuse of notation
        e_X = X[i] + 1;
        y = sqrt(e_X * e_X);
        X_graded[i] = X[i]/(2 * y) + 1;
      }
      return;
    default:
      break;
  }
  
  switch (strcmp(function, "sinus")) {
    case 0:
      for (i = threshold; i--; ) {
        X_graded[i] = cos(X[i]);
      }
      return;
    default:
      break;
  }
  
  switch (strcmp(function, "sinusc")) {
    case 0:
      for (i = threshold; i--; ) {
        switch (X[i] == 0.0) {
          case 1:
            X_graded[i] = 0.0;
            break;
          default:
            X_graded[i] = cos(X[i])/X[i] - sin(X[i])/(X[i] * X[i]);
            break;
        }
      }
      return;
    default:
      break;
  }
  
  // Gaussian if all else fails
  for (i = threshold; i--; ) {
    X_graded[i] = -2 * X[i] * exp(-(X[i] * X[i]));
  }
}
