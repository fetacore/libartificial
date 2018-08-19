#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

void row_sum(double *row_sum, double *matrix, int rows, int columns) {
	
	int i;
	
	// vector of ones to get average delta (to sum deltas over all rows)
	double *ones = malloc(rows * sizeof(double));
	for (i = 0; i < rows; i++) {
		ones[i] = 1.0;
	}
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
							1, // Rows of z[0][l][i][j]
							columns, // Columns of z[0][l][i][j]
							rows, // columns of A, rows of B
							1.0, // scaling factor (none)
							ones, rows, // C = A * B -> matrix A, ldA
							matrix, columns, // C = A * B -> matrix B, ldB
							1.0, // scaling factor for C
							row_sum, columns); // C, ldC
	
	free(ones);
}
