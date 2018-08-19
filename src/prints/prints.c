#include <stdio.h>
#include <stdlib.h>
#include "../headers/utils.h"

int l, i, j;

void showXY(int rows, int columns, double X[rows][columns], double Y[rows]) {
	printf("My initial X \n");
	for (i = 0; i < rows; i++) {
		for (j = 0; j < columns; j++) {
			printf("%f\t", X[i][j]);
		}
		printf("\n");
	}
	for (i = 0; i < rows; i++) {
		printf("%f\n", Y[i]);
	}
	printf("\n");
}

void showNormalized(int rows, int columns, double *X) {
	printf("\n was normalized\n");
	for (i = 0; i < rows; i++) {
		for (j = 0; j < columns; j++) {
			printf("%f\t", X[i * columns + j]);
		}
		printf("\n");
	}
}

void showWB(int layers, int nodes[layers], int columns_Y, int columns_X, double ***wb) {
	int o, l, i, j;
	for (o = 0; o < 2; o++) {
		if (o == 0) {
			// 			Weights
			printf("\nWeights...\n");
			for (l = 0; l < layers + 1; l++) {
				if (l == 0) {
					printf("\t\tyInput layer\n");
					for (i = 0; i < columns_X; i++) {
						for (j = 0; j < nodes[l]; j++) {
							printf("%f\t", wb[o][l][i * nodes[l] + j]);
						}
						printf("\n");
					}
					printf("\n");
				} else if (l == layers) {
					printf("\t\tOutput layer\n");
					for (i = 0; i < nodes[l-1]; i++) {
						for (j = 0; j < columns_Y; j++) {
							printf("%f\t", wb[o][l][i * columns_Y + j]);
						}
						printf("\n");
					}
				} else {
					printf("\t\tHidden Layer no.%d\n", l);
					for (i = 0; i < nodes[l-1]; i++) {
						for (j = 0; j < nodes[l]; j++) {
							printf("%f\t", wb[o][l][i * nodes[l] + j]);
						}
						printf("\n");
					}
					printf("\n");
				}
				printf("\n");
			}
		} else {
			// 			Biases
			printf("Biases...\n");
			for (l = 0; l < layers + 1; l++) {
				if (l == 0) {
					printf("\t\tInput layer\n");
					for (j = 0; j < nodes[0]; j++) {
						printf("%f\t", wb[o][l][j]);
						// 						printf("%f\t", biases[l][0][j]);
					}
					printf("\n");
				} else if (l == layers) {
					for (j = 0; j < columns_Y; j++) {
						printf("\tOutput layer\n%f\t", wb[o][l][j]);
					}
					printf("\n");
				} else {
					printf("\tLayer no.%d\n", l);
					for (j = 0; j < nodes[l]; j++) {
						printf("%f\t", wb[o][l][j]);
						// 						printf("%f\t", biases[l][0][j]);
					}
					printf("\n");
				}
			}
		}
	}
}
