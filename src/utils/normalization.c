#include <stdio.h>
#include <math.h>

double mean(const size_t rows, double col[rows]) {
  double sum = 0.0, out;
  for (size_t i = 0; i < rows; i++) {
    sum += col[i];
  }
  out = sum/rows;
  return out;
}

double stdev(const size_t rows, double col[rows], const double mean) {
  double sumsq = 0.0, out, subtr;
  for (size_t i = 0; i < rows; i++) {
    subtr = col[i] - mean;
    sumsq += subtr * subtr;
  }
  out = sqrt(sumsq/(rows - 1));
  return out;
}

// Takes an x matrix with each column representing different variables
// and each row representing values of each variable
void normalize(double *X, const size_t rows, const size_t columns) {
  size_t i, j;
  double m = 0.0, sd = 0.0, col[rows];
  
  for (j = 0; j < columns; j++) {
    for (i = 0; i < rows; i++) {
      col[i] = X[i * columns + j];
    }
    m = mean(rows, col);
    sd = stdev(rows, col, m);
    for (i = 0; i < rows; i++) {
      X[i * columns + j] = (X[i * columns + j] - m)/sd;
    }
    m = 0.0;
    sd = 0.0;
  }
}
