#include <stdio.h>
#include <stdlib.h>

int ***imgpad(int ***images, int no_of_images, int img_width, int img_height, int img_channels,
              int padding, // Zeros around
              int delete_originals // 0 if no, 1 if yes (keep only vectorized in memory)
             )
{
  if (padding == 0) {
    return images;
  } else {
    int image, i, j, rgb;
    int multiplication = (img_width + 2 * padding) * (img_height + 2 * padding);
    
    int ***images_padded = malloc(no_of_images * sizeof(int **));
    
    for (image = no_of_images; image--; ) {
      images_padded[image] = malloc(multiplication * sizeof(int *));
      for (i = multiplication; i--; ) {
        images_padded[image][i] = malloc(img_channels * sizeof(int));
        for (rgb = img_channels; rgb--; ) {
          images_padded[image][i][rgb] = 0;
        }
      }
      for (i = img_height; i--; ) {
        for (j = img_width; j--; ) {
          for (rgb = img_channels; rgb--; ) {
            images_padded[image][(i + padding) * (img_width + 2 * padding) + j + padding][rgb] =
                                                                                images[image][i * img_width + j][rgb];
          }
        }
      }
    }
    
    if (delete_originals == 1) {
      multiplication = img_width * img_height;
      for (i = 0; i < no_of_images; i++) {
        for (j = 0; j < multiplication; j++) {
          free(images[i][j]);
        }
        free(images[i]);
      }
      free(images);
    }
    return images_padded;
  }
}


int **im2col(int ***images, int no_of_images, int img_width, int img_height, int img_channels,
             int spatial, // width and height of weights
             int stride, // (img_width - spatial + 2 * padding)/stride should be int
             int padding, // Zeros around
             int delete_originals // 0 if no, 1 if yes (keep only vectorized in memory)
            )
{
  int image, i, i_prime, j, pixel_x, pixel_y, rgb, multiplication;
  
  // How many boxes horizontally
  int locations_width = (img_width - spatial + 2 * padding)/stride + 1;
  // How many boxes vertically
  int locations_height = (img_height - spatial + 2 * padding)/stride + 1;
  
  // Rows of vectorized image
  int receptive_fields = locations_width * locations_height;
  
  multiplication = spatial * spatial * img_channels;
  
  int ***images_w_pad = imgpad(images, no_of_images, img_width, img_height, img_channels, padding, delete_originals);
  
  int **images_as_matrices = malloc(no_of_images * sizeof(int *));
  for (image = no_of_images; image--; ) {
    
    pixel_x = 0;
    pixel_y = 0;
    i_prime = 0;
    
    // dim(receptive_fields X (spatial * spatial * img_channels))
    images_as_matrices[image] = malloc(receptive_fields * multiplication * sizeof(int));
    
    for (i = 0; i < receptive_fields; i++) {
      rgb = 0;
      if (i % locations_height == 0 && i > 0) {
        i_prime += 1;
        pixel_x = 0;
      } else if (i % locations_height != 0 && i > 0){
        pixel_x += stride;
      } else {
        pixel_x = 0;
      }
      
      pixel_y = i_prime * stride;
      
      for (j = 0; j < multiplication; j++) {
        
        // The channel change happens in every row for every conv box
        if (j > 0 && j % (spatial * spatial) == 0) {
          rgb += 1;
          if (rgb == img_channels) {
            rgb = 0;
          }
        }
        
        // Width and height pixels
        if (j % spatial == 0 && j > 0) {
          pixel_y += 1;
          if (j % (spatial * spatial) == 0) {
            pixel_y = i_prime * stride;
          }
          
          pixel_x -= spatial - 1;
          if (i % locations_height == 0 && i > 0) {
            pixel_x = 0;
          }
          
        } else if (j % spatial != 0 && j > 0) {
          pixel_x += 1;
        }
        
        /////////////////////////////////////////////////////////////
        // The operation
        images_as_matrices[image][i * multiplication + j] =
                                                images_w_pad[image][pixel_x * (img_width + 2 * padding) + pixel_y][rgb];
        if (j == multiplication - 1) {
          pixel_x -= spatial - 1;
        }
      }
    }
  }
  
  if (padding != 0) {
    multiplication = (img_width + 2 * padding) * (img_height + 2 * padding);
    for (i = 0; i < no_of_images; i++) {
      for (j = 0; j < multiplication; j++) {
        free(images_w_pad[i][j]);
      }
      free(images_w_pad[i]);
    }
    free(images_w_pad);
  }
  
  if (delete_originals == 1 && padding == 0) {
    for (i = 0; i < no_of_images; i++) {
      for (j = 0; j < img_width * img_height; j++) {
        free(images[i][j]);
      }
      free(images[i]);
    }
    free(images);
  }
  
  return images_as_matrices;
}
