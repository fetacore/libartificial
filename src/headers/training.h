#ifndef training_h__
#define training_h__

extern void update_gd(int rows, int columns_Y, int columns_X, int batch,
											int layers, int nodes[layers],
											double *Y, double *X, double ***Z, double ***wb,
											char funcs[layers+1][30], double learning_rate,
											int epochs);

extern void delta_gd(double **deltas, int rows, int columns_Y, int layers,
										 double *Y_row, double ***Z_row, double ***wb,
										 int nodes[layers], char funcs[layers+1][30]);

// While updating we need to find Z with the new weights n biases
extern void feedforward_update(double ***Z,
															int rows,
															int columns_Y,
															int columns_X,
															int layers,
															double *X, double ***wb,
															int nodes[layers],
															char funcs[layers+1][30]);
#endif  // training_h__
