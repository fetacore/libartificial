void row_sum(double *row_sum, double *matrix, int rows, int columns) {
  
  int i = 0, j = 0, threshold = rows * columns;
  
  while (i != threshold) {
    if (j > columns) {
      j = 0;
    }
    row_sum[j++] += matrix[i++];
  }
}
