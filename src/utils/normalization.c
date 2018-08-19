#include <stdio.h>
#include <math.h>

double mean(int rows, double col[rows]) {
  double sum = 0.0, out;
  for (int i = 0; i < rows; i++) {
    sum += col[i];
  }
  out = sum/rows;
  return out;
}

double stdev(int rows, double col[rows], double mean) {
  double sumsq = 0.0, out, subtr;
  for (int i = 0; i < rows; i++) {
    subtr = col[i] - mean;
    sumsq += subtr * subtr;
  }
  out = sqrt(sumsq/(rows - 1));
  return out;
}

// Takes an x matrix with each column representing different variables
// and each row representing values of each variable
void normalize(double *X, int rows, int columns) {
	int i, j;
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
