#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double rmse(int rows, int columns_Y, double *Y, double *prediction) {
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
