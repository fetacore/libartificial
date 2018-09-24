#include <stdlib.h>
#include <math.h>
#include <time.h>

double rand_normal(const double mu, const double sigma) {//Box muller method
  
  static double n2 = 0.0;
  static double n2_cached = 0.0;
  if (!n2_cached) {
    double x, y, r;
    do
    {
      x = 2.0 * (double)rand()/RAND_MAX - 1;
      y = 2.0 * (double)rand()/RAND_MAX - 1;
      
      r = x*x + y*y;
    } while (r == 0.0 || r > 1.0);
    double d = sqrt(-2.0 * log(r)/r);
    double n1 = x * d;
    n2 = y*d;
    double result = n1 * sigma + mu;
    n2_cached = 1.0;
    return result;
  } else {
    n2_cached = 0.0;
    return n2 * sigma + mu;
  }
}
