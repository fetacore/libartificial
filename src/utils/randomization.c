#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// A utility function to swap to integers
void swap(double *a, double *b)
{
	int temp = *a;
	*a = *b;
	*b = temp;
}

// A function to generate a random permutation of inputs
void randomize(double *X, int rows, int columns_X)
{
	// Use a different seed value so that we don't get same
	// result each time we run this program
	srand(time(NULL));
	
	// Start from the last element and swap one by one. We don't
	// need to run for the first element that's why i > 0
	for (int i = rows * columns_X - 1; i > 0; i--) {
		// Pick a random index from 0 to i
		int j = rand() % (i+1);
		
		// Swap X[i] with the element at random index
		swap(&X[i], &X[j]);
	}
}
