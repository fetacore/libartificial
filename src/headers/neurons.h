#ifndef neurons_h__
#define neurons_h__

extern double ***feedforward(int rows, int columns_Y, int columns_X, int layers, double *X, double ***wb,
                             int nodes[layers], char funcs[layers+1][30]);

#endif  // neurons_h__
