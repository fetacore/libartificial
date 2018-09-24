#include <stdlib.h>

void row_sum(double *row_sum, double *matrix, const int rows, const int columns)
{
  int i = rows * columns - 1;
  int j = columns - 1;
  
  do {
    if (i > (rows - 1) * columns - 1) {
      row_sum[j] = 0;
    }
    row_sum[j] += matrix[i];
    if (--j < 0) {
      j = columns - 1;
    }
  } while (--i >= 0);
}
