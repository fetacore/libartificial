#include <stdlib.h>

void row_sum(double *row_sum, const double *matrix,
             const size_t rows, const size_t columns)
{
  size_t i = 0, j = 0;
  const size_t threshold = rows * columns;
  
  while (i != threshold) {
    if (j == columns) {
      j = 0;
    }
    row_sum[j++] += matrix[i++];
  }
}
