#include <math.h>
#include <stdio.h>

double rmse(const int *restrict rows, const int *restrict columns_Y,
            const double *restrict Y, const double *restrict prediction) {
  int i = (*rows) * (*columns_Y) - 1;
  double loss = 0.0;
  double dif;
  
  do {
    dif = prediction[i] - Y[i];
    loss += (dif * dif)/(double)(*columns_Y);
  } while (--i >= 0);
  
  loss = loss/(double)(*rows);
  return sqrt(loss);
}

double xentropy(const int *restrict rows, const int *restrict columns_Y,
                const double *restrict Y, const double *restrict prediction) {
  int i = (*rows) * (*columns_Y) - 1;
  double loss = 0.0;
  
  do {
    loss += Y[i] * log(prediction[i])/(double)(*columns_Y);
  } while (--i >= 0);
  return -loss/(double)(*rows);
}
