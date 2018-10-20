/*
 * libartificial - Small header-only C library for Artificial Neural Networks
 * 
 * Copyright (c) 2018 Jim Karoukis
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */



#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../libartificial.h"

int main(void) {
  
  srand(time(NULL));
  const int training_images = 40000;
  const int test_images = 400;
  const int width = 7; // pixels_x (columns)
  const int height = 5; // pixels_y (rows)
  // RGB = 3 dimensions
  // RGB range = 0 -> 255
  const int depth = 3; // channels (color or bnw)
  const int labels = 10;
  
  
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
//   const int filters = 2;
  const int spatial_extent = 2;
  const int stride = 1;
  const int padding = 1;
//   const int dilation = 0;
//   const double w_variance = 0.01;
  ///////////////////////////////////////////////////////////////////////////
  // User specific
  
  // After vectorization
  const int delete_originals = 1;
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
  
  int **train_imgs_vector = im2col(imgdata_train, &training_images, &width, &height,
                                   &depth, &spatial_extent, &stride, &padding, &delete_originals);
//   int **test_imgs_vector = im2col(imgdata_test, &test_images, &width, &height,
//                                   &depth, &spatial_extent, &stride, &padding, &delete_originals);
  
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
  delete_img_vector(&training_images, train_imgs_vector);
  //   delete_img_vector(&test_images, test_imgs_vector);
  free(training_label_values);
  free(test_label_values);
  return 0;
}
