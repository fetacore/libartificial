#include <math.h>

double rmse(const int rows, const int columns_Y, const double *Y, const double *prediction) {
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
