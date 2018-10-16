#include <stdlib.h>
#include <string.h>

int *name2int(const int layers, char funcs[layers+1][30]) {
  
  int *names2ints = malloc((layers + 1) * sizeof(int));
  int l;
  
  #pragma omp parallel for
  for (l = 0; l < layers + 1; l++) {
    switch (strcmp(funcs[l], "relu")) {
      case 0:
        names2ints[l] = 0;
        continue;
      default:
        break;
    }
    
    switch (strcmp(funcs[l], "logistic")) {
      case 0:
        names2ints[l] = 1;
        continue;
      default:
        break;
    }
    
    switch (strcmp(funcs[l], "linear")) {
      case 0:
        names2ints[l] = 2;
        continue;
      default:
        break;
    }
    
    switch (strcmp(funcs[l], "tanh")) {
      case 0:
        names2ints[l] = 3;
        continue;
      default:
        break;
    }
    
    switch (strcmp(funcs[l], "softmax")) {
      case 0:
        names2ints[l] = 4;
        continue;
      default:
        break;
    }
    
    // Leaky relu
    switch (strcmp(funcs[l], "lrelu")) {
      case 0:
        names2ints[l] = 5;
        continue;
      default:
        break;
    }
    
    switch (strcmp(funcs[l], "softplus")) {
      case 0:
        names2ints[l] = 6;
        continue;
      default:
        break;
    }
    
    switch (strcmp(funcs[l], "softsign")) {
      case 0:
        names2ints[l] = 7;
        continue;
      default:
        break;
    }
    
    switch (strcmp(funcs[l], "arctan")) {
      case 0:
        names2ints[l] = 8;
        continue;
      default:
        break;
    }
    
    //Inverse square root with a = 1
    switch (strcmp(funcs[l], "isru")) {
      case 0:
        names2ints[l] = 9;
        continue;
      default:
        break;
    }
    
    //Inverse sqrt linear unit \w a=1
    switch (strcmp(funcs[l], "isrlu")) {
      case 0:
        names2ints[l] = 10;
        continue;
      default:
        break;
    }
    
    switch (strcmp(funcs[l], "bent")) {
      case 0:
        names2ints[l] = 11;
        continue;
      default:
        break;
    }
    
    switch (strcmp(funcs[l], "sinus")) {
      case 0:
        names2ints[l] = 12;
        continue;
      default:
        break;
    }
    
    switch (strcmp(funcs[l], "sinusc")) {
      case 0:
        names2ints[l] = 13;
        continue;
      default:
        break;
    }
    
    // Gaussian if nothing else
    names2ints[l] = 14;
  }
  
  return names2ints;
}
