#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double rmse(const size_t rows, const size_t columns_Y, const double *Y, const double *prediction) {
  int i, thresh = rows * columns_Y;
  double loss = 0.0;
  double dif;
  
  for (i = 0; i < thresh; i++) {
    dif = prediction[i] - Y[i];
    loss += (dif * dif)/(double)columns_Y;
  }
  loss = loss/(double)rows;
  return sqrt(loss);
}
