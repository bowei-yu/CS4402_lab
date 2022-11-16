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
// double ** alloc_matrix(int n, int m);
int MPI_Is_sorted(int n, double * array, int * answer, int root, MPI_Comm comm);
int MPI_Exchange( int n, double * array, int rank1, int rank2, MPI_Comm comm);

int isActive( int rank, int p, int l );
int isReciever( int rank, int p, int l );
int isSender( int rank, int p, int l );

int MPI_Sort_direct(int n, double * a, int root, MPI_Comm comm);
int MPI_Sort_bucket(int n, double * a, double m, int root, MPI_Comm comm);
int MPI_Sort_shell(int n, double * a, int root, MPI_Comm comm);
int MPI_Sort_bitonic(int n, double * a, int root, MPI_Comm comm);
int MPI_Sort_oddeven( int n, double * array, int root, MPI_Comm comm );
// int MPI_Sort_bucketalltoall(int n, double * a, double m, int root, MPI_Comm comm);
// mergesort implementation is in another .c file

int main (int argc, char *argv[])
{

	int rank, size;

	int n = 10000000, i, j, k, x, q, l, shell, pair, *nr;
	double m = 10.0;
	double * array;

  MPI_Status status;

	// Init + rank + size
	MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  array = (double *) calloc( n, sizeof(double) );

	if( rank == 0 )
	{

	   //initialise the array with random values, then scatter to all processors
	    srand( ((unsigned)time(NULL)+rank) );

        for( i = 0; i < n; i++ )
        {
            array[i]=((double)rand()/RAND_MAX)*m;
            // printf("%lf ", array[i]);
        }

	}

    // call and time evaluate MPI_Sort
    double time = MPI_Wtime();
    double overallTime;

    MPI_Sort_bucket( n, array, m, 0, MPI_COMM_WORLD );
    // MPI_Sort_oddeven( n, array, 0, MPI_COMM_WORLD );
    // MPI_Sort_bitonic( n, array, 0, MPI_COMM_WORLD );
    // MPI_Sort_shell( n, array, 0, MPI_COMM_WORLD );
    // MPI_Sort_direct( n, array, 0, MPI_COMM_WORLD );
    
    time = MPI_Wtime() - time;
    MPI_Reduce(&time, &overallTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if(rank == 0) {
        // printf("Output:\n");
        // for(int i = 0; i < n; i++) printf("%lf ", array[i]);
        printf("\nExecution Time with %d procs is %lf\n", size, overallTime);
    }
    

	MPI_Finalize();

}

// MPI Functions

int MPI_Sort_bitonic(int n, double * a, int root, MPI_Comm comm) {
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double * localArray = (double *) calloc (n/size, sizeof(double));

    double TComm, TComp = 0;
    double time = MPI_Wtime();

    int rc = MPI_Scatter(a, n / size, MPI_DOUBLE, localArray, n / size, MPI_DOUBLE, root, comm);
    if(rc != MPI_SUCCESS) return rc;

    time = MPI_Wtime() - time;
    TComm += time;

    time = MPI_Wtime();
    merge_sort(n/size, localArray);
    time = MPI_Wtime() - time;
    TComp += time;

    time = MPI_Wtime();
    for (int d = 0; d < log2(size); d++) {
      for (int i = d; i >= 0; i--) {
        int pair = rank ^ (1 << i);
        if (((rank >> (d + 1)) % 2 == 0 && (rank >> i) % 2 == 0) || ((rank >> (d + 1)) % 2 != 0 && (rank >> i) % 2 != 0)) {
          MPI_Exchange(n/size, localArray, rank, pair, MPI_COMM_WORLD);
        } else {
          MPI_Exchange(n/size, localArray, pair, rank, MPI_COMM_WORLD);
        }
      }
    }

    rc = MPI_Gather(localArray, n / size, MPI_DOUBLE, a, n / size, MPI_DOUBLE, root, comm);
    if(rc != MPI_SUCCESS) return rc;

    time = MPI_Wtime() - time;
    TComm += time;

    printf("Exec time on proc %d: TComm %f TComp %f, TComm + TComp: %f\n", rank, TComm, TComp, TComm + TComp);

    return MPI_SUCCESS;
}

int MPI_Sort_shell(int n, double * a, int root, MPI_Comm comm) {

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double * localArray = (double *) calloc (n/size, sizeof(double));

    double TComp, TComm = 0;
    double time = MPI_Wtime();

    int rc = MPI_Scatter(a, n / size, MPI_DOUBLE, localArray, n / size, MPI_DOUBLE, root, comm);
    if(rc != MPI_SUCCESS) return rc;

    time = MPI_Wtime() - time;
    TComm += time;

    // sort localArray
    time = MPI_Wtime();
    merge_sort(n / size, localArray);
    time = MPI_Wtime() - time;
    TComp += time;

    time = MPI_Wtime();
    int left = 0, right = size - 1;
    while (left < right) {
        int pair = right + left - rank;
        if (rank < pair) {
            MPI_Exchange(n/size, localArray, rank, pair, comm);

        }
        if (rank > pair) {
            MPI_Exchange(n/size, localArray, pair, rank, comm);
        }
        // half
        int mid = (left + right) / 2;
        if (mid < rank) left = mid + 1;
        else right = mid;
    }

    //odd-even iterations
    for (int i = 0;i < size; i++) {
        if ( (i+rank)%2 ==0 ){
            if ( rank < size-1 ) MPI_Exchange(n/size, localArray,rank,rank+1,MPI_COMM_WORLD);
        } else {
            if( rank > 0 ) MPI_Exchange(n/size,localArray,rank-1,rank,MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // test if overall array is sorted and break if required
        int answer;
        // gather all first and last of each array from each processor and do a comparison
        MPI_Is_sorted(n/size, localArray, &answer, root, comm);

        if (answer == 1) {
          if (rank == root) {
            printf("Odd even finished in %d operations\n", i + 1);
          }
          break;
        }
    }

    rc = MPI_Gather(localArray, n / size, MPI_DOUBLE, a, n / size, MPI_DOUBLE, root, comm);
    if(rc != MPI_SUCCESS) return rc;

    time = MPI_Wtime() - time;
    TComm += time;

    printf("Exec time on proc %d: TComm %f TComp %f, TComm + TComp: %f\n", rank, TComm, TComp, TComm + TComp);

    return MPI_SUCCESS;
}

int MPI_Sort_oddeven( int n, double * array, int root, MPI_Comm comm ) {
      
  // get rank and size of comm
  int rank, size;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  
  //allocate space for numElements/numProcessors amount of doubles
  double * localArray = ( double * ) calloc( n / size, sizeof( double ) );

  // manage the time
  double TComm = 0, TComp = 0;
  double time = MPI_Wtime();
    
  //scatter a to local_a
  int rc = MPI_Scatter(array, n / size, MPI_DOUBLE, localArray, n / size, MPI_DOUBLE, root, comm);
  if(rc != MPI_SUCCESS) return rc;
  
  time = MPI_Wtime() - time;  
  TComm += time;
  
  //sort local_a using mergeSort
  time = MPI_Wtime();  
  merge_sort(n/size, localArray);
  time =  MPI_Wtime() - time;
  TComp += time;
    
  //odd-even iterations
  time = MPI_Wtime(); 
  for (int i = 0; i < size; i++) {
    if ( (i+rank)%2 ==0 ){
        if ( rank < size-1 ) MPI_Exchange(n/size, localArray,rank,rank+1,MPI_COMM_WORLD);
    } else {
        if( rank > 0 ) MPI_Exchange(n/size,localArray,rank-1,rank,MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // test if overall array is sorted and break if required
    int answer;
    // gather all first and last of each array from each processor and do a comparison
    MPI_Is_sorted(n/size, localArray, &answer, root, comm);

    if (answer == 1) {
      if (rank == root) {
        printf("Odd even finished in %d operations\n", i + 1);
      }
      break;
    }
  }
  
  //gather local_a
  rc = MPI_Gather( localArray, n/size, MPI_DOUBLE, array, n/size, MPI_DOUBLE, 0, MPI_COMM_WORLD );
  if(rc != MPI_SUCCESS) return rc;

  time =  MPI_Wtime() - time;
  TComm += time;

  printf("Exec time on proc %d: TComm %f TComp %f, TComm + TComp: %f\n", rank, TComm, TComp, TComm + TComp);

  return MPI_SUCCESS;
  
}

int MPI_Is_sorted(int n, double * array, int * answer, int root, MPI_Comm comm) {
  // get rank and size
  int rank, size;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &size );

  // allocate first and last with size elements
  double * first = (double *) calloc(size, sizeof(double));
  double * last = (double*) calloc(size, sizeof(double));
  
  // gather first and last
  MPI_Gather(&array[0], 1, MPI_DOUBLE, first, 1, MPI_DOUBLE, root, comm);
  MPI_Gather(&array[n-1], 1, MPI_DOUBLE, last, 1, MPI_DOUBLE, root, comm);
  
  // if rank is root then test
  if (rank == root) {
    *answer = 1;
    for (int i = 1; i < size; i++) {
      if (first[i] < last[i - 1]) { 
        // first of cuurent processor is smaller than last of previous processor
        // means not sorted, return 0
        *answer = 0; break;
      }
    }
  }

  // bcast the answer
  MPI_Bcast(answer, 1, MPI_INT, root, comm);

  return MPI_SUCCESS;
}

int MPI_Sort_bucket(int n, double * a, double m, int root, MPI_Comm comm) {
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // manage the time
    double TComm = 0, TComp = 0;
    double time =  MPI_Wtime();
    
    int rc = MPI_Bcast(a, n, MPI_DOUBLE, root, comm);
    if (rc != MPI_SUCCESS) return rc;
    
    time = MPI_Wtime() - time;
    TComm += time;

    // worst case: all n elements in bucket
    double * bucket = (double * ) calloc(n, sizeof(double));
    int count = 0;

    time =  MPI_Wtime();
    
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
    TComp += time;

    // gatherv buckets ==> gather recvCounts, calc displacement
    int * recvCounts = (int *) calloc(size, sizeof(int));
    int * displacement = (int *) calloc(size, sizeof(int));

    time =  MPI_Wtime();

    rc = MPI_Gather(&count, 1, MPI_INT, recvCounts, 1, MPI_INT, root, comm);
    if (rc != MPI_SUCCESS) return rc;

    time = MPI_Wtime() - time;
    TComm += time;
    
    if (rank == 0) {
        time = MPI_Wtime();
        displacement[0] = 0;
        for (int i = 1; i < size; i++) {
            displacement[i] = displacement[i - 1] + recvCounts[i - 1];
        }
        time = MPI_Wtime() - time;
        TComp += time;
    }
    
    time =  MPI_Wtime();
    
    rc = MPI_Gatherv(bucket, count, MPI_DOUBLE, a, recvCounts, displacement, MPI_DOUBLE, root, comm);
    if (rc != MPI_SUCCESS) return rc;
    
    time = MPI_Wtime() - time;
    TComm += time;

    printf("Exec time on proc %d: TComm %f TComp %f, TComm + TComp: %f\n", rank, TComm, TComp, TComm + TComp);

    return MPI_SUCCESS;

}

int MPI_Sort_direct(int n, double * array, int root, MPI_Comm comm){
    
   int rank, size;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);

   double * localArray = (double *) calloc (n/size, sizeof(double));

   double TComm, TComp = 0;
   double time = MPI_Wtime();

   // scatter array to localArray with double elements
   int rc = MPI_Scatter(array, n / size, MPI_DOUBLE, localArray, n / size, MPI_DOUBLE, root, comm);
   if(rc != MPI_SUCCESS) return rc;

   time = MPI_Wtime();
   // sort localArray
   merge_sort(n / size, localArray);
   time = MPI_Wtime() - time;
   TComp += time;

   // gather localArray to array with double elements
   time = MPI_Wtime();
   rc = MPI_Gather(localArray, n / size, MPI_DOUBLE, array, n / size, MPI_DOUBLE, root, comm);
   if(rc != MPI_SUCCESS) return rc;
   time = MPI_Wtime() - time;
   TComm += time;

   if( rank == 0 )
   {
        // merge the size chunks of array
        time = MPI_Wtime();
        for(int i = 1; i < size; i++) {
            double * tmp = merge_array(i * n / size, array, n / size, array + i * n / size);
            for(int j=0; j < (i + 1) * n / size; j++) array[j] = tmp[j];
        }
        time = MPI_Wtime() - time;
        TComp += time;
   }

   printf("Exec time on proc %d: TComm %f TComp %f, TComm + TComp: %f\n", rank, TComm, TComp, TComm + TComp);

   return MPI_SUCCESS;

}

// will be worked on after
// int MPI_Sort_bucketalltoall(int n, double * a, double m, int root, MPI_Comm comm) {

//   int rank, size;
//   MPI_Comm_rank(comm, &rank);
//   MPI_Comm_size(comm, &size);

//   double* localArray = (double *) calloc(n/size, sizeof(double));
//   int* countersj = (int *) calloc(size, sizeof(int));
//   for (int i = 0; i < size; i++) countersj[i] = 0;

//   int rc = MPI_Scatter(a, n / size, MPI_DOUBLE, localArray, n / size, MPI_DOUBLE, root, comm);
//   if(rc != MPI_SUCCESS) return rc;

//   for (int i = 0; i < n/size; i++) {
//     printf("%f\n", localArray[i]);
//   }

//   for (int i = 0; i < n/size; i++) {
//     for (int j = 0; j < size; j++) {
//       // check if a[i] is in correct bucket
//       if (j * m/size <= localArray[i] && localArray[i] < (j + 1) * m/size) {
//           printf("%f-%d:B:%dC-%d  ", localArray[i], rank, j, countersj[j]);
//           buckets[j][countersj[j]++] = localArray[i];
//           // printf("%f:%f - %f ", localArray[i], j * m/size, (j + 1) * m/size);
//           break;
//       }
//     } 
//   }

//   int *global_bucket_count = (int*)calloc(P*sizeof(int));
// 	MPI_Alltoall(countersj, 1, MPI_INT, global_bucket_count, 1, MPI_INT, MPI_COMM_WORLD);

//   for (int j = 0; j < size; j++) {
//     merge_sort(countersj[j], buckets[j]);
//     for (int i = 0; i < countersj[j]; i++) printf("%f-->R%d:B%d ", buckets[j][i], rank, j);

//     MPI_Alltoallv(buckets[j], countersj[j], MPI_DOUBLE, localArray,
//                  int count_recv,
//                  MPI_Datatype datatype_recv,
//                  MPI_Comm communicator);
//   }

//   MPI_Alltoall(const void* buffer_send,
//                  int count_send,
//                  MPI_Datatype datatype_send,
//                  void* buffer_recv,
//                  int count_recv,
//                  MPI_Datatype datatype_recv,
//                  MPI_Comm communicator);

//   // // gatherv buckets ==> gather recvCounts, calc displacement
//   // int * recvCounts = (int *) calloc(size, sizeof(int));
//   // int * displacement = (int *) calloc(size, sizeof(int));

//   //   time =  MPI_Wtime();
//   //   MPI_Gather(&count, 1, MPI_INT, recvCounts, 1, MPI_INT, root, comm);
//   //   time = MPI_Wtime() - time;
//   //   tcomm += time;
    
//   //   if (rank == 0) {
//   //       displacement[0] = 0;
//   //       for (int i = 1; i < size; i++) {
//   //           displacement[i] = displacement[i - 1] + recvCounts[i - 1];
//   //       }
//   //   }
//   //   time =  MPI_Wtime();
//   //   MPI_Gatherv(bucket, count, MPI_DOUBLE, a, recvCounts, displacement, MPI_DOUBLE, root, comm);
//   //   time = MPI_Wtime() - time;
//   //   tcomm += time;

//   //   printf("Exec time on proc %d: comm %f comp %f, total: %f\n", rank, tcomm, tcomp, tcomm + tcomp);

//   //   return MPI_SUCCESS;



//   return MPI_SUCCESS;

// }

// double ** alloc_matrix(int n, int m){

// 	int i, j;
//   double **a, *aa;

// 	aa=(double *) calloc(n*m, sizeof(double));
// 	a=(double **) calloc(n, sizeof(double*));

// 	for(i=0;i<n;i++)a[i]=aa+i*m;

// 	for(i=0;i<n;i++)for(j=0;j<m;j++)a[i][j]=0;

// 	return a;
// }

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

//notes
    double * merge( int n, double * a, int m, double * b ) {
       int i, j, k;
       double * c = ( double * ) calloc( n + m, sizeof( double ) );
     
       for( i=j=k=0; ( i < n ) && ( j < m ); )
       {
          if( a[ i ] <= b[ j ] )
          {
            c[ k++ ] = a[ i++ ];
          }
          else
          {
            c[ k++ ] = b[ j++ ];
          }
       }
      if( i == n )
      {
        for( ; j < m; )
        {
          c[ k++ ] = b[ j++ ];
        }
      }
      else
      {
        for( ; i < n; )
        {
          c[ k++ ] = a[ i++ ];
        }
      }
      return c;
    }

int MPI_Exchange( int n, double * array, int rank1, int rank2, MPI_Comm comm ) {
      int rank, size, result, i, tag1 = 0, tag2 = 1;
      double * b = ( double * ) calloc( n, sizeof( double ) );
      double * c;
       
      MPI_Status status;
      MPI_Comm_rank( comm, &rank );
      MPI_Comm_size( comm, &size );
     
      //L8.6
      if( rank == rank1 )
      {
        result = MPI_Send( &array[ 0 ], n, MPI_DOUBLE, rank2, tag1, comm );
        result = MPI_Recv( &b[ 0 ], n, MPI_DOUBLE, rank2, tag2, comm, &status );
        c = merge( n, array, n, b );
        for( i = 0; i < n; i++ )
        {
          array[ i ] = c[ i ];
        }
      }
      else if( rank == rank2 )
      {
        result = MPI_Recv( &b[ 0 ], n, MPI_DOUBLE, rank1, tag1, comm, &status );
        result = MPI_Send( &array[ 0 ], n, MPI_DOUBLE, rank1, tag2, comm) ;
        c = merge( n, array, n, b );
        for( i =0; i < n; i++ )
        {
          array[ i ] = c[ i + n ];
        }
      }
      return MPI_SUCCESS;
    }