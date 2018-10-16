#include <stdlib.h>

void delete_Z(const int layers, double ***Z)
{
  int l;
  for (l = 0; l < layers + 1; l++) {
    free(Z[1][l]);
    free(Z[0][l]);
  }
  free(Z[1]);
  free(Z[0]);
  free(Z);
}

void delete_w(const int layers, double **w)
{
  int l;
  for (l = 0; l < layers + 1; l++) {
    free(w[l]);
  }
  free(w);
}

void delete_img_vector(int **images, const size_t no_of_images)
{
  size_t image;
  
  for (image = 0; image < no_of_images; image++) {
    free(images[image]);
  }
  free(images);
}
