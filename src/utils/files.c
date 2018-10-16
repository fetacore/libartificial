#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

void save_w(double **w, const int layers, const int nodes[layers],
             const int columns_Y, const int columns_X) {
  
  int l;
  
  char path[1024];
  char w_path[1036];
  getcwd(path, sizeof(path));
  strcpy(w_path, path);
  
  struct stat st = {0};
  if (stat(strcat(w_path, "/weights"), &st) == -1) {
    mkdir(w_path, 0700);
    printf("\nCreated ./wb/weights directory\n");
  }
  
  FILE *ptr_fp;
  
  for (l = 0; l < layers + 1; l++) {
    
    // This needs to change every time
    char w_path_filename[1050];
    char number[100];
    char filename[15] = "layer_";
    sprintf(number, "%d", l);
    strcat(filename, number);
    strcat(filename, ".bin");
    
    strcpy(w_path_filename, w_path);
    strcat(w_path_filename, "/");
    
    strcat(w_path_filename, filename);
    
    if((ptr_fp = fopen(w_path_filename, "wb")) == NULL) {
			printf("Unable to create file!\n");
			exit(1);
		}
    
    if (l == 0) {
      if(fwrite(w[l], columns_X * nodes[l] * sizeof(double), 1, ptr_fp) != 1) {
        printf("Write error!\n");
        exit(1);
      }
      fclose(ptr_fp);
    } else if (l == layers) {
      if(fwrite(w[l], nodes[l-1] * columns_Y * sizeof(double), 1, ptr_fp) != 1) {
        printf("Write error!\n");
        exit(1);
      }
      fclose(ptr_fp);
    } else {
      if(fwrite(w[l], nodes[l-1] * nodes[l] * sizeof(double), 1, ptr_fp) != 1) {
        printf("Write error!\n");
        exit(1);
      }
      fclose(ptr_fp);
    }
  }
}

double **load_w(const int layers, const int nodes[layers], const int columns_Y, const int columns_X) {
  
  int l;
  
  char path[1024];
  getcwd(path, sizeof(path));
  char w_path[1036];
  strcpy(w_path, path);
  strcat(w_path, "/weights");
  
  double **w = malloc((layers + 1) * sizeof(double *));
  
  w[0] = malloc(columns_X * nodes[0] * sizeof(double));  
  switch (layers > 1) {
    case 1:
      for (l = 1; l < layers; l++) {
        w[l] = malloc(nodes[l-1] * nodes[l] * sizeof(double));
      }
      break;
    default:
      break;
  }
  w[layers] = malloc(nodes[layers - 1] * columns_Y * sizeof(double));
  
  FILE *ptr_fp;
  
  for (l = 0; l < layers + 1; l++) {
    
    // This needs to change every time
    char w_path_filename[1050];
    char number[100];
    char filename[15] = "layer_";
    sprintf(number, "%d", l);
    strcat(filename, number);
    strcat(filename, ".bin");
    
    strcpy(w_path_filename, w_path);
    strcat(w_path_filename, "/");
    strcat(w_path_filename, filename);
    
    if((ptr_fp = fopen(w_path_filename, "rb")) == NULL) {
			printf("Unable to open file!\n");
			exit(1);
		}
    
    if (l == 0) {
      if(fread(w[l], columns_X * nodes[l] * sizeof(double), 1, ptr_fp) != 1) {
        printf("Read error!\n");
        exit(1);
      }
      fclose(ptr_fp);
    } else if (l == layers) {
      if(fread(w[l], nodes[l-1] * columns_Y * sizeof(double), 1, ptr_fp) != 1) {
        printf("Read error!\n");
        exit(1);
      }
      fclose(ptr_fp);
    } else {
      if(fread(w[l], nodes[l-1] * nodes[l] * sizeof(double), 1, ptr_fp) != 1) {
        printf("Read error!\n");
        exit(1);
      }
      fclose(ptr_fp);
    }
  }
  return w;
}
