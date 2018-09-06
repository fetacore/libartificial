#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

void save_wb(double ***wb, int layers, int nodes[layers], int columns_Y, int columns_X) {
  
  int l;
  
  char path[1024];
  char w_path[1036];
  char b_path[1035];
  getcwd(path, sizeof(path));
  strcpy(w_path, path);
  strcpy(b_path, path);
  
  struct stat st = {0};
  if (stat(strcat(w_path, "/wb"), &st) == -1) {
    mkdir(w_path, 0700);
  }
  
  if (stat(strcat(w_path, "/weights"), &st) == -1) {
    mkdir(w_path, 0700);
    printf("\nCreated wb/weights directory\n");
  }
  
  if (stat(strcat(b_path, "/wb"), &st) == -1) {
    mkdir(b_path, 0700);
  }
  
  if (stat(strcat(b_path, "/biases"), &st) == -1) {
    mkdir(b_path, 0700);
    printf("\nCreated wb/biases directory\n");
  }
  
  FILE *ptr_fp;
  
  for (l = 0; l < layers + 1; l++) {
    
    // This needs to change every time
    char w_path_filename[1050];
    char b_path_filename[1050];
    char number[100];
    char filename[15] = "layer_";
    sprintf(number, "%d", l);
    strcat(filename, number);
    strcat(filename, ".data");
    
    strcpy(w_path_filename, w_path);
    strcpy(b_path_filename, b_path);
    strcat(w_path_filename, "/");
    strcat(b_path_filename, "/");
    
    strcat(w_path_filename, filename);
    strcat(b_path_filename, filename);
        
    if((ptr_fp = fopen(w_path_filename, "wb")) == NULL) {
			printf("Unable to create file!\n");
			exit(1);
		}
    
    if (l == 0) {
      if(fwrite(wb[0][l], columns_X * nodes[l] * sizeof(double), 1, ptr_fp) != 1) {
        printf("Write error!\n");
        exit(1);
      }
      fclose(ptr_fp);
    } else if (l == layers) {
      if(fwrite(wb[0][l], nodes[l-1] * columns_Y * sizeof(double), 1, ptr_fp) != 1) {
        printf("Write error!\n");
        exit(1);
      }
      fclose(ptr_fp);
    } else {
      if(fwrite(wb[0][l], nodes[l-1] * nodes[l] * sizeof(double), 1, ptr_fp) != 1) {
        printf("Write error!\n");
        exit(1);
      }
      fclose(ptr_fp);
    }
    
    if((ptr_fp = fopen(b_path_filename, "wb")) == NULL) {
			printf("Unable to create file!\n");
			exit(1);
		}
		
    if (l == 0) {
      if(fwrite(wb[1][l], nodes[l] * sizeof(double), 1, ptr_fp) != 1) {
        printf("Write error!\n");
        exit(1);
      }
      fclose(ptr_fp);
    } else if (l == layers) {
      if(fwrite(wb[1][l], columns_Y * sizeof(double), 1, ptr_fp) != 1) {
        printf("Write error!\n");
        exit(1);
      }
      fclose(ptr_fp);
    } else {
      if(fwrite(wb[1][l], nodes[l] * sizeof(double), 1, ptr_fp) != 1) {
        printf("Write error!\n");
        exit(1);
      }
      fclose(ptr_fp);
    }
		
  }
    
}

double ***load_wb(int layers, int nodes[layers], int columns_Y, int columns_X) {
  
  int l;
  
  char path[1024];
  getcwd(path, sizeof(path));
  char w_path[1036];
  char b_path[1035];
  strcpy(w_path, path);
  strcpy(b_path, path);
  
  strcat(w_path, "/wb/weights");
  strcat(b_path, "/wb/biases");
  
  double ***wb = malloc(2 * sizeof(double **));
  wb[0] = malloc((layers + 1) * sizeof(double *));
  wb[1] = malloc((layers + 1) * sizeof(double *));
  
  wb[0][0] = malloc(columns_X * nodes[0] * sizeof(double));
  wb[1][0] = malloc(nodes[0] * sizeof(double));
  
  switch (layers > 1) {
    case 1:
      for (l = 1; l < layers; l++) {
        wb[0][l] = malloc(nodes[l-1] * nodes[l] * sizeof(double));
        wb[1][l] = malloc(nodes[l] * sizeof(double));
      }
      break;
    default:
      break;
  }
  
  wb[0][layers] = malloc(nodes[layers - 1] * columns_Y * sizeof(double));
  wb[1][layers] = malloc(columns_Y * sizeof(double));
  
  FILE *ptr_fp;
  
  for (l = 0; l < layers + 1; l++) {
    
    // This needs to change every time
    char w_path_filename[1050];
    char b_path_filename[1050];
    char number[100];
    char filename[15] = "layer_";
    sprintf(number, "%d", l);
    strcat(filename, number);
    strcat(filename, ".data");
    
    strcpy(w_path_filename, w_path);
    strcpy(b_path_filename, b_path);
    strcat(w_path_filename, "/");
    strcat(b_path_filename, "/");
    
    strcat(w_path_filename, filename);
    strcat(b_path_filename, filename);
    
    if((ptr_fp = fopen(w_path_filename, "rb")) == NULL) {
			printf("Unable to open file!\n");
			exit(1);
		}
    
    if (l == 0) {
      if(fread(wb[0][l], columns_X * nodes[l] * sizeof(double), 1, ptr_fp) != 1) {
        printf("Read error!\n");
        exit(1);
      }
      fclose(ptr_fp);
    } else if (l == layers) {
      if(fread(wb[0][l], nodes[l-1] * columns_Y * sizeof(double), 1, ptr_fp) != 1) {
        printf("Read error!\n");
        exit(1);
      }
      fclose(ptr_fp);
    } else {
      if(fread(wb[0][l], nodes[l-1] * nodes[l] * sizeof(double), 1, ptr_fp) != 1) {
        printf("Read error!\n");
        exit(1);
      }
      fclose(ptr_fp);
    }
    
    if((ptr_fp = fopen(b_path_filename, "rb")) == NULL) {
			printf("Unable to open file!\n");
			exit(1);
		}
		
    if (l == 0) {
      if(fread(wb[1][l], nodes[l] * sizeof(double), 1, ptr_fp) != 1) {
        printf("Write error!\n");
        exit(1);
      }
      fclose(ptr_fp);
    } else if (l == layers) {
      if(fread(wb[1][l], columns_Y * sizeof(double), 1, ptr_fp) != 1) {
        printf("Write error!\n");
        exit(1);
      }
      fclose(ptr_fp);
    } else {
      if(fread(wb[1][l], nodes[l] * sizeof(double), 1, ptr_fp) != 1) {
        printf("Write error!\n");
        exit(1);
      }
      fclose(ptr_fp);
    }
		
  }
  
  return wb;
}
