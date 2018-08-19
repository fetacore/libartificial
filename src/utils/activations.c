#include <string.h>
#include <math.h>

int i;

void activate(double *X_activated, double *X, int threshold, char *function) {
	
	switch (strcmp(function, "relu")) {
		case 0:
			for (i = threshold; i--; ) {
				switch (X[i] < 0.0) {
					case 1:
						X_activated[i] = 0.0;
						break;
					default:
						X_activated[i] = X[i];
						break;
				}
			}
			return;
		default:
			break;
	}
	
	// Leaky relu
	switch (strcmp(function, "lrelu")) {
		case 0:
			for (i = threshold; i--; ) {
				switch (X[i] < 0.0) {
					case 1:
						X_activated[i] = 0.01 * X[i];
						break;
					default:
						X_activated[i] = X[i];
						break;
				}
			}
			return;
		default:
			break;
	}
	
	double e_X = 0, e_mX = 0;
	switch (strcmp(function, "tanh")) {
		case 0:
			for (i = threshold; i--; ) {
				X_activated[i] = tanh(X[i]);
			}
			return;
		default:
			break;
	}
	
	switch (strcmp(function, "logistic")) {
		case 0:
			for (i = threshold; i--; ) {
				e_mX = exp(-X[i]);
				X_activated[i] = 1/(1 + e_mX);
			}
			return;
		default:
			break;
	}
	
	switch (strcmp(function, "linear")) {
		case 0:
			// Check here
			for (i = threshold; i--; ) {
				X_activated[i] = X[i];
			}
			return;
		default:
			break;
	}
	
	switch (strcmp(function, "softmax")) {
		case 0:
			for (i = threshold; i--; ) {
				// Abuse of notation (sum of all exp(X_i))
				e_X += exp(X[i]);
			} 
			for (i = threshold; i--; ) {
				X_activated[i] = exp(X[i])/e_X;
			}
			return;
		default:
			break;
	}
	
	switch (strcmp(function, "softplus")) {
		case 0:
			for (i = threshold; i--; ) {
				e_X = exp(X[i]);
				X_activated[i] = log(1 + e_X);
			}
			return;
		default:
			break;
	}
	
	switch (strcmp(function, "softsign")) {
		case 0:
			for (i = threshold; i--; ) {
				X_activated[i] = X[i]/(1 + fabs(X[i]));
			}
			return;
		default:
			break;
	}
	
	switch (strcmp(function, "arctan")) {
		case 0:
			for (i = threshold; i--; ) {
				X_activated[i] = atan(X[i]);
			}
			return;
		default:
			break;
	}
	
	//Inverse square root with a = 1
	switch (strcmp(function, "isru")) {
		case 0:
			for (i = threshold; i--; ) {
				X_activated[i] = X[i]/sqrt(1 + pow(X[i], 2));
			}
			return;
		default:
			break;
	}
	
	//Inverse sqrt linear unit \w a=1
	switch (strcmp(function, "isrlu")) {
		case 0:
			for (i = threshold; i--; ) {
				switch (X[i] < 0.0) {
					case 1:
						X_activated[i] = X[i]/sqrt(1 + pow(X[i], 2));
						break;
					default:
						X_activated[i] = X[i];
						break;
				}
			}
			return;
		default:
			break;
	}
	
	switch (strcmp(function, "bent")) {
		case 0:
			for (i = threshold; i--; ) {
				X_activated[i] = (sqrt(pow(X[i], 2) + 1.0) - 1.0)/2.0 + X[i];
			}
			return;
		default:
			break;
	}
	
	switch (strcmp(function, "sinus")) {
		case 0:
			for (i = threshold; i--; ) {
				X_activated[i] = sin(X[i]);
			}
			return;
		default:
			break;
	}
	
	switch (strcmp(function, "sinusc")) {
		case 0:
			for (i = threshold; i--; ) {
				switch (X[i] == 0.0) {
					case 1:
						X_activated[i] = 1.0;
						break;
					default:
						X_activated[i] = sin(X[i])/X[i];
						break;
				}
			}
			return;
		default:
			break;
	}
	
	switch (strcmp(function, "gauss")) {
		case 0:
			for (i = threshold; i--; ) {
				X_activated[i] = exp(-pow(X[i], 2));
			}
			return;
		default:
			// Last of them all just return x
			X_activated = X;
			return;
	}
}
