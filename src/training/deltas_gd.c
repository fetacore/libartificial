#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>

#include "../headers/utils.h"

// DELTAS FOR EVERY TYPE OF GRADIENT DESCENT
// (batch => rows = all rows)
// (stochastic => rows = 1)
// (mini-batch => rows = rows of mini set)

void delta_gd(double **deltas, int rows, int columns_Y, 
							int layers, double *Y, double ***Z, double ***wb,
							int nodes[layers], char funcs[layers+1][30])
{
	int l, i;
	int for_helper = rows * columns_Y;
	// Deltas filling
	//////////////////////////////////////////////////////////////////
	// Gradient of layer's unactivated output
	double **help_1 = malloc(layers * sizeof(double *));
	// Product of next layer's transposed weights and deltas
	double **help_2 = malloc(layers * sizeof(double *));
	// We do not need them at the output layer
	//////////////////////////////////////////////////////////////////
	
	// That is for the last layer's delta (computed first)
	
// 	for (i = 0; i < for_helper; i++) {
// 		switch (strcmp(funcs[layers], "linear")) {
// 			case 0:
// 				deltas[layers][i] = Z[1][layers][i] - Y[i];
// 				break;
// 			default:
// 				deltas[layers][i] = gradient(Z[0][layers][i], funcs[layers]) *
// 														(Z[1][layers][i] - Y[i]);
// 				break;
// 		}
// 	}
	
	switch (strcmp(funcs[layers], "linear")) {
		case 0:
			for (i = 0; i < for_helper; i++) {
				deltas[layers][i] = Z[1][layers][i] - Y[i];
			}
			break;
		default:
			gradient(deltas[layers], Z[0][layers], for_helper, funcs[layers]);
			break;
	}
	
	for (l = layers - 1; l >= 0; l--) {
		for_helper = rows * nodes[l];
		
		help_1[l] = malloc(for_helper * sizeof(double));
		help_2[l] = malloc(for_helper * sizeof(double));
		
		gradient(help_1[l], Z[0][l], for_helper, funcs[l]);
		for (i = 0; i < for_helper; i++) {
			help_2[l][i] = 0.0;
		}
		
		switch (l == layers - 1) {
			// True
			case 1:
				cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
										rows, // Rows of help_2[0][j]
										nodes[l], // Columns of help_2[0][j]
										columns_Y, // columns of A, rows of B
										1.0, // scaling factor (none)
										deltas[l+1], columns_Y, // C = A * B -> matrix A, ldA
										wb[0][l+1], columns_Y, // C = A * B -> matrix B, ldB
										1.0, // scaling factor for C (none)
										help_2[l], nodes[l]); // C, ldC
				
				// Hadamard product
				for (i = 0; i < for_helper; i++) {
					deltas[l][i] = help_1[l][i] * help_2[l][i];
				}
				
				free(help_2[l]);
				free(help_1[l]);
				
				break;
			// False
			default:
				// find help_2
				cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
										rows, // Rows of help_2[0][j]
										nodes[l], // Columns of help_2[0][j]
										nodes[l+1], // columns of A, rows of B
										1.0, // scaling factor (none)
										deltas[l+1], nodes[l+1], // C = A * B -> matrix A, ldA
										wb[0][l+1], nodes[l+1], // C = A * B -> matrix B,  ldB
										1.0, // scaling factor for C (none)
										help_2[l], nodes[l]); // C, ldC
				
				// Hadamard product
				for (i = 0; i < for_helper; i++) {
					deltas[l][i] = help_1[l][i] * help_2[l][i];
				}
				
				free(help_2[l]);
				free(help_1[l]);
				
				break;
		}
	}
	free(help_2);
	free(help_1);
}
