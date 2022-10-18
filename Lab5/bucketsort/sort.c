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
int MPI_Sort_bucket(int n, double * a, double m, int root, MPI_Comm comm);

int main (int argc, char *argv[])
{

	int rank, size;

	int n = 10000000, i, j, k, x, q, l, shell, pair, *nr;
	double m = 10.0;
	double * scattered_array, * array;

	// Init + rank + size
	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    array = (double *) calloc( n, sizeof(double) );

	if( rank == 0 )
	{

	   //initialise the array with random values, then scatter to all processors
        //array = (double *) calloc( n, sizeof(double) );
	    srand( ((unsigned)time(NULL)+rank) );

        for( i = 0; i < n; i++ )
        {
            array[i]=((double)rand()/RAND_MAX)*m;
        }

	}

    // call and time evaluate MPI_Sort_direct
    double time = MPI_Wtime();
    double overallTime;
    MPI_Sort_bucket(n, array, m, 0, MPI_COMM_WORLD);
    time = MPI_Wtime() - time;
    MPI_Reduce(&time, &overallTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        // for(int i = 0; i < n; i++) printf("%lf \n", array[i]);
        printf("Execution Time with %d procs is %lf", size, overallTime);
    }
    

	MPI_Finalize();

}

// MPI Functions

int MPI_Sort_bucket(int n, double * a, double m, int root, MPI_Comm comm) {
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // manage the time
    double tcomm = 0, tcomp = 0;

    double time =  MPI_Wtime();
    MPI_Bcast(a, n, MPI_DOUBLE, root, comm);
    time = MPI_Wtime() - time;
    tcomm += time;

    // worst case: all n elements in bucket
    double * bucket = (double * ) calloc(n, sizeof(double));
    int count = 0;

    // scan array and collect into bucket
    for (int i = 0; i < n; i ++) {
        // check if a[i] is in bucket rank
        if (rank * m/size <= a[i] && a[i] < (rank + 1) * m/size) {
            bucket[count++] = a[i];
        }
    }

    // sort bucket
    merge_sort(count, bucket);
    time = MPI_Wtime() - time;
    tcomp += time;

    // gatherv buckets ==> gather recvCounts, calc displacement
    int * recvCounts = (int *) calloc(size, sizeof(int));
    int * displacement = (int *) calloc(size, sizeof(int));

    time =  MPI_Wtime();
    MPI_Gather(&count, 1, MPI_INT, recvCounts, 1, MPI_INT, root, comm);
    time = MPI_Wtime() - time;
    tcomm += time;
    
    if (rank == 0) {
        displacement[0] = 0;
        for (int i = 1; i < size; i++) {
            displacement[i] = displacement[i - 1] + recvCounts[i - 1];
        }
    }
    time =  MPI_Wtime();
    MPI_Gatherv(bucket, count, MPI_DOUBLE, a, recvCounts, displacement, MPI_DOUBLE, root, comm);
    time = MPI_Wtime() - time;
    tcomm += time;

    printf("Exec time on proc %d: comm %lf comp %lf\n", rank, tcomm, tcomp);

    return MPI_SUCCESS;

}

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