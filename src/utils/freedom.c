#include <stdio.h>
#include <stdlib.h>

void delete_Z(int layers, double ***Z)
{
  int o, l;
  for (o = 0; o < 2; o++) {
    for (l = 0; l < layers + 1; l++) {
      free(Z[o][l]);
    }
    free(Z[o]);
  }
  free(Z);
}

void delete_wb(int layers, double ***wb)
{
  int l;
  for (l = 0; l < layers + 1; ++l) {
    free(wb[1][l]);
    free(wb[0][l]);
  }
  free(wb[1]);
  free(wb[0]);
  free(wb);
}
