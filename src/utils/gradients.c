#include <string.h>
#include <math.h>

void gradient(double *X_graded, const double *X, const int threshold, char *function) {
  
  int i = threshold - 1;
  double e_X = 0.0, e_mX = 0.0, y = 0.0;
  
  switch (strcmp(function, "relu")) {
    case 0:
      do {
        switch (X[i] < 0.0) {
          case 1:
            X_graded[i--] = 0.0;
            continue;
          default:
            X_graded[i--] = 1.0;
            continue;
        }
      } while (i >= 0);
      return;
    default:
      break;
  }
  
  switch (strcmp(function, "logistic")) {
    case 0:
      do {
        e_mX = exp(-X[i]);
        y = 1/(1 + e_mX);
        X_graded[i--] = y * (1 - y);
      } while (i >= 0);
      return;
    default:
      break;
  }
  
  switch (strcmp(function, "linear")) {
    case 0:
      do {
        X_graded[i--] = 1.0;
      } while (i >= 0);
      return;
    default:
      break;
  }
  
  switch (strcmp(function, "tanh")) {
    case 0:
      do {
        e_X = exp(X[i]);
        e_mX = exp(-X[i]);
        y = (e_X - e_mX)/(e_X + e_mX);
        X_graded[i--] = 1 - y * y;
      } while (i >= 0);
      return;
    default:
      break;
  }
  
  
  switch (strcmp(function, "softmax")) {
    case 0:
      do {
        // Abuse of notation (sum of all exp(X_i))
        e_X += exp(X[i--]);
      } while (i >= 0);
      
      i = threshold - 1;
      
      do {
        // Abuse again
        e_mX = exp(X[i])/e_X;
        X_graded[i--] = e_mX * (1 - e_mX);
      } while (i >= 0);
      return;
    default:
      break;
  }
  
  // Leaky relu
  switch (strcmp(function, "lrelu")) {
    case 0:
      do {
        switch (X[i] < 0.0) {
          case 1:
            X_graded[i--] = 0.01;
            continue;
          default:
            X_graded[i--] = 1.0;
            continue;
        }
      } while (i >= 0);
      return;
    default:
      break;
  }
  
  switch (strcmp(function, "softplus")) {
    case 0:
      do {
        e_mX = exp(-X[i]);
        X_graded[i--] = 1/(1 + e_mX);
      } while (i >= 0);
      return;
    default:
      break;
  }
  
  switch (strcmp(function, "softsign")) {
    case 0:
      do {
        // abuse of notation
        e_X = fabs(X[i]);
        y = 1 + e_X;
        X_graded[i--] = 1/(y * y);
      } while (i >= 0);
      return;
    default:
      break;
  }
  
  switch (strcmp(function, "arctan")) {
    case 0:
      do {
        // Abuse of notation
        e_X = X[i] * X[i];
        X_graded[i--] = 1/(e_X + 1);
      } while (i >= 0);
      return;
    default:
      break;
  }
  
  //Inverse square root with a = 1
  switch (strcmp(function, "isru")) {
    case 0:
      do {
        y = sqrt(1 + X[i] * X[i]);
        y = X[i]/y;
        X_graded[i--] = pow(y, 3);
      } while (i >= 0);
      return;
    default:
      break;
  }
  
  //Inverse sqrt linear unit \w a=1
  switch (strcmp(function, "isrlu")) {
    case 0:
      do {
        switch (X[i] < 0.0) {
          case 1:
            y = sqrt(1 + X[i] * X[i]);
            y = X[i]/y;
            X_graded[i--] = pow(y, 3);
            continue;
          default:
            X_graded[i--] = 1.0;
            continue;
        }
      } while (i >= 0);
      return;
    default:
      break;
  }
  
  switch (strcmp(function, "bent")) {
    case 0:
      do {
        // abuse of notation
        e_X = X[i] + 1;
        y = sqrt(e_X * e_X);
        X_graded[i] = X[i]/(2 * y) + 1;
        i--;
      } while (i >= 0);
      return;
    default:
      break;
  }
  
  switch (strcmp(function, "sinus")) {
    case 0:
      do {
        // Abuse of notation
        e_X = cos(X[i]);
        X_graded[i--] = e_X;
      } while (i >= 0);
      return;
    default:
      break;
  }
  
  switch (strcmp(function, "sinusc")) {
    case 0:
      do {
        switch (X[i] == 0.0) {
          case 1:
            X_graded[i--] = 0.0;
            continue;
          default:
            X_graded[i] = cos(X[i])/X[i] - sin(X[i])/(X[i] * X[i]);
            i--;
            continue;
        }
      } while (i >= 0);
      return;
    default:
      break;
  }
  
  // Gaussian if all else fails
  do {
    X_graded[i] = -2 * X[i] * exp(-(X[i] * X[i]));
    i--;
  } while (i >= 0);
}
