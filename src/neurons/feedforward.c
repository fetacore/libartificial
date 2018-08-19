#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

#include "../headers/utils.h"
#include "../headers/training.h"
#include "../headers/prints.h"

double ***feedforward(int rows, int columns_Y, int columns_X, int layers,
											double *X, double ***wb,
											int nodes[layers], char funcs[layers+1][30])
{
	// l is for layers
	// i is for each row * column of X, Y
	int l, i;
	
	// feeds at every layer
	double ***Z = malloc(2 * sizeof(double **));
	Z[0] = malloc((layers + 1) * sizeof(double *));
	Z[1] = malloc((layers + 1) * sizeof(double *));
	for (l = 0; l < layers + 1; l++) {
		switch (l == layers) {
			// Statement true
			case 1:
				Z[0][l] = malloc(rows * columns_Y * sizeof(double));
				Z[1][l] = malloc(rows * columns_Y * sizeof(double));
				for (i = 0; i < rows * columns_Y; i++) {
					Z[0][l][i] = 0.0;
					Z[1][l][i] = 0.0;
				}
				break;
			default:
				Z[0][l] = malloc(rows * nodes[l] * sizeof(double));
				Z[1][l] = malloc(rows * nodes[l] * sizeof(double));
				for (i = 0; i < rows * nodes[l]; i++) {
					Z[0][l][i] = 0.0;
					Z[1][l][i] = 0.0;
				}
				break;
		}
	}
	
	// Directly manipulates Z
	feedforward_update(Z, rows, columns_Y, columns_X, layers, X,wb, nodes, funcs);
	
	return Z;
}
