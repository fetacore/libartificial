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
  double log_pred = 0.0;
  
  do {
    log_pred = log(prediction[i]);
    loss += (Y[i] * log_pred + (1.0 - Y[i]) * (1.0 - log_pred))/(double)(*columns_Y);
  } while (--i >= 0);
  
  loss = loss/(double)(*rows);
  return loss;
}
