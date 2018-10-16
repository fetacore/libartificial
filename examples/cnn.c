#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../src/headers/utils.h"

#define training_images 40000
#define test_images 400
#define width 7 // pixels_x (columns)
#define height 5 // pixels_y (rows)
// RGB = 3 dimensions
// RGB range = 0 -> 255
#define depth 3 // channels (color or bnw)
#define labels 10

int main(void) {
  
  srand(time(NULL));
  
  int i, j, rgb, random_prob_one;
  
  // Training set (Our X in training)
  int ***imgdata_train = malloc(training_images * sizeof(int **));
  for (i = 0; i < training_images; i++) {
    imgdata_train[i] = malloc(width * height * sizeof(int *));
    for (j = 0; j < width * height; j++) {
      imgdata_train[i][j] = malloc(depth * sizeof(int));
      for (rgb = 0; rgb < depth; rgb++) {
        imgdata_train[i][j][rgb] = rand() % (255 + 1 - 0) + 0; // range(0, 255)
      }
    }
  }
  
  // Testing set (Our X in testing)
  int ***imgdata_test = malloc(test_images * sizeof(int **));
  for (i = 0; i < test_images; i++) {
    imgdata_test[i] = malloc(width * height * sizeof(int *));
    for (j = 0; j < width * height; j++) {
      imgdata_test[i][j] = malloc(depth * sizeof(int));
      for (rgb = 0; rgb < depth; rgb++) {
        imgdata_test[i][j][rgb] = rand() % (255 + 1 - 0) + 0; // range(0, 255)
      }
    }
  }
  
  // Probability of training image belonging to specific label (Our Y in training)
  double *training_label_values = malloc(training_images * labels * sizeof(double));
  for (i = 0; i < training_images; i++) {
    // Chooses randomly where to put the 1.0 probability and all else 0
    random_prob_one = rand() % ((labels - 1) + 1 - 0) + 0; // range(0, labels)
    for (j = 0; j < labels; j++) {
      if (j == random_prob_one) {
        training_label_values[i * labels + j] = 1.0;
      } else {
        training_label_values[i * labels + j] = 0.0;
      }
    }
  }
  
  // Probability of test image belonging to specific label (Our Y in testing)
  double *test_label_values = malloc(test_images * labels * sizeof(double));
  for (i = 0; i < test_images; i++) {
    // Chooses randomly where to put the 1.0 probability and all else 0
    random_prob_one = rand() % ((labels - 1) + 1 - 0) + 0; // range(0, labels)
    for (j = 0; j < labels; j++) {
      if (j == random_prob_one) {
        test_label_values[i * labels + j] = 1.0;
      } else {
        test_label_values[i * labels + j] = 0.0;
      }
    }
  }
  
  ///////////////////////////////////////////////////////////////////////////
  // Hyperparameters
  ///////////////////////////////////////////////////////////////////////////
  int filters = 2;
  int spatial_extent = 2;
  int stride = 1;
  int padding = 1;
  int dilation = 0;
  double w_variance = 0.01;
  ///////////////////////////////////////////////////////////////////////////
  // User specific
  
  // After vectorization
  int delete_originals = 1;
  ///////////////////////////////////////////////////////////////////////////
  
  // How many layers is #conv operations X filters
//   int conv_operations = 1; // One convolution operation for now
//   int layers = conv_operations * filters;
  // nodes[layers]
//   int *nodes = malloc(layers * sizeof(int));
//   for (i = 0; i < layers; i++) {
//     nodes[i] = spatial_extent;
//   }
  ///////////////////////////////////////////////////////////////////////////
  
  // Weight initialization
  //                                 #layers,
//   double ***wb = init_wb(w_variance, layers,  )
  
  // Neuron
  /*
  printf("\n");
  for (rgb = 0; rgb < depth; rgb++) {
    printf("RGB = %d\n", rgb);
    for (j = 0; j < height; j++) {
      for (i = 0; i < width; i++) {
        printf("%d\t", imgdata_train[1][j * width + i][rgb]);
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("\n\n");
  */
  
  // Training
  
  int **train_imgs_vector = im2col(imgdata_train, training_images, width, height, depth, spatial_extent, stride, padding,
                                   delete_originals);
//   int **test_imgs_vector = im2col(imgdata_test, test_images, width, height, depth, spatial_extent, stride, padding,
//                                   delete_originals);
  
  // Freedom
  if (delete_originals == 0) {
    for (i = 0; i < training_images; i++) {
      for (j = 0; j < width * height; j++) {
        free(imgdata_train[i][j]);
      }
      free(imgdata_train[i]);
    }
    free(imgdata_train);
    
    for (i = 0; i < test_images; i++) {
      for (j = 0; j < width * height; j++) {
        free(imgdata_test[i][j]);
      }
      free(imgdata_test[i]);
    }
    free(imgdata_test);
  }
  delete_img_vector(train_imgs_vector, training_images);
//   delete_img_vector(test_imgs_vector, test_images);
  free(training_label_values);
  free(test_label_values);
  return 0;
}
