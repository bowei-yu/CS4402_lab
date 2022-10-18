/***************************************************************


 MPI program to compute an array operation

****************************************************************/



#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "limits.h"


int main(int argc, char * argv []) {


	int * scattered_array, * array;
	int n = 10000000; int n1;
	int size, rank;

	// declare and initialise the variables for sum, prod, max and min
	int sum = 0, final_sum = 0, prod = 0, final_prod = 0, max = 0, final_max = 0, min = INT_MAX, final_min = INT_MAX;

	// Init + rank + size
	MPI_Init (&argc,&argv);
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);

	scattered_array=(int *)calloc(n/size, sizeof(int));

	// 	populate the array a if processor 0
	// 	serial code
	if (rank == 0) {

	   // Processor 0 is the only one to know array
	   array = (int *)calloc(n, sizeof(int));
	   unsigned int iseed = (unsigned int) time(NULL);
	   for(int i=0;i<n;i++) {
		srand(iseed);
		array[i] = rand() % 2; // you can initialise with a random number
		printf(array[i]);
	   }
	}
	
	time_start = MPI_Wtime();
	// scatter the array onto processors
	// local array is scattered_array
	MPI_Scatter(array, n/size, MPI_INT, scattered_array, n/size, MPI_INT, 0, MPI_COMM_WORLD);


	// calculate sum, prod, max and min of scattered_array
	for(int i=0;i<n/size;i++)
	{
		max = scattered_array[i] > max ? scattered_array[i] : max;
		min = scattered_array[i] < min ? scattered_array[i] : min;
		sum += scattered_array[i];
		prod = prod * scattered_array[i];
	}
	
	double time_end = MPI_Wtime();
	double time = time_end - time_start;
	double overall_time = 0;

	// reduce sum, prod, max, min to final_sum, final_prod, final_max and final_min
	MPI_Reduce(&sum, &final_sum, 1, MPI_INT,MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&prod, &final_prod, 1, MPI_INT, MPI_PROD, 0, MPI_COMM_WOLRD);
	MPI_Reduce(&max, &final_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&min, &final_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD); 
	MPI_Reduce(&time, &overall_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); 
	
	if(rank==0){

	// Write final_sum, final_prod, final_max and final_min

	   printf("the final sum is %d", final_sum);
	   printf("the final prod is %d", final_prod);
	   printf("the final max is %d", final_max);
	   printf("the final min is %d", final_min);
	   printf("the overall execution time is %f", &overall_time); 
	}

	MPI_Finalize ();

}
