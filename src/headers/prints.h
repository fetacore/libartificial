#ifndef prints_h__
#define prints_h__

extern void showXY(int rows, int columns, double X[rows][columns], double Y[rows]);
extern void showNormalized(int rows, int columns, double *X);
extern void showWB(int layers, int nodes[layers], int columns_Y, int columns_X, double ***wb);

#endif  // prints_h__
