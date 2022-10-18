/**
 * MPI Program that performs simple sorting
 *
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"



double * merge_array(int n, double * a, int m, double * b);
void     merge_sort(int n, double * a);
void     swap (double * a, double * b);
void     bubble_sort(int n, double * a);

int MPI_Sort_direct(int n, double * a, int root, MPI_Comm comm);


int main (int argc, char *argv[])
{

	int rank, size;

	int n = 10000, i, j, k, x, q, l, shell, pair, *nr;
	double m = 10.0;
	double * scattered_array, * array;

	// Init + rank + size
	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


	if( rank == 0 )
	{

	   //initialise the array with random values, then scatter to all processors
        array = (double *) calloc( n, sizeof(double) );
	    srand( ((unsigned)time(NULL)+rank) );

        for( i = 0; i < n; i++ )
        {
            array[i]=((double)rand()/RAND_MAX)*m;
        }

	}

    // call and time evaluate MPI_Sort_direct
    double time = MPI_Wtime();
    double overallTime;
    MPI_Sort_direct(n, array, 0, MPI_COMM_WORLD);
    time = MPI_Wtime() - time;
    MPI_Reduce(&time, &overallTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        for(int i = 0; i < n; i++) printf("%lf \n", array[i]);
        printf("Execution Time with %d procs is %lf", size, overallTime);
    }
    

	MPI_Finalize();

}

// MPI Functions

int MPI_Sort_direct(int n, double * array, int root, MPI_Comm comm){
    
   int rank, size;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);

   double * localArray = (double *) calloc (n/size, sizeof(double));

   // scatter array to localArray with double elements
   int rc = MPI_Scatter(array, n / size, MPI_DOUBLE, localArray, n / size, MPI_DOUBLE, root, comm);
   if(rc != MPI_SUCCESS) return rc;

   // sort localArray
   merge_sort(n / size, localArray);


   // gather localArray to array with double elements
   rc = MPI_Gather(localArray, n / size, MPI_DOUBLE, array, n / size, MPI_DOUBLE, root, comm);
   if(rc != MPI_SUCCESS) return rc;




   if( rank == 0 )
   {
        // merge the size chunks of array
        for(int i = 1; i < size; i++) {
            double * tmp = merge_array(i * n / size, array, n / size, array + i * n / size);
            for(int j=0; j < (i + 1) * n / size; j++) array[j] = tmp[j];
        }

        // print array


   }

   return MPI_SUCCESS;

}

// function to merge the array a with n elements with the array b with m elements
// function returns the nerged array

double * merge_array(int n, double * a, int m, double * b){

   int i,j,k;
   double * c = (double *) calloc(n+m, sizeof(double));

   for(i=j=k=0;(i<n)&&(j<m);)

      if(a[i]<=b[j])c[k++]=a[i++];
      else c[k++]=b[j++];

   if(i==n)for(;j<m;)c[k++]=b[j++];
   else for(;i<n;)c[k++]=a[i++];

return c;
}

// function to merge sort the array a with n elements

void merge_sort(int n, double * a){

   double * c;
   int i;

   if (n<=1) return;

   if(n==2) {

      if(a[0]>a[1])swap(&a[0],&a[1]);
      return;
   }



   merge_sort(n/2,a);merge_sort(n-n/2,a+n/2);

   c=merge_array(n/2,a,n-n/2,a+n/2);

   for(i=0;i<n;i++)a[i]=c[i];

return;
}


// swap two doubles
void swap (double * a, double * b){

   double temp;

   temp=*a;*a=*b;*b=temp;

}


void bubble_sort(int n, double * array) {
    for(int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n-1; j++) {
            if(array[j] > array[j + 1]) {
                swap(array+j, array+j+1);
            }
        }
    }
}