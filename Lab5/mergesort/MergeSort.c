#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

int isActive( int rank, int p, int l );
int isReciever( int rank, int p, int l );
int isSender( int rank, int p, int l );

double * merge_array(int n, double * a, int m, double * b);
void merge_sort(int n, double * a);
void swap (double * a, double * b);

int main (int argc, char *argv[])
{

	int rank, size;

	int n = 10000000, q, level, i, j, k, x, *nr;
	double m = 10.0;
	double *a, *b;

	MPI_Status status;

	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	a = (double *) calloc(n,sizeof(double));
	b = (double *) calloc(n,sizeof(double));

	if( rank == 0 )
	{

	   //initialise the array with random values, then scatter to all processors
	   srand( ((unsigned)time(NULL)+rank) );

        // printf("Initial: \n");
	   for( x = 0; x < n; x++ )
	   {
	      a[x]=((double)rand()/RAND_MAX)*m;
	    //   printf( "%f ", a[x] );
	   }

	   printf("Sorting %d elements with %d processors\n", n, size);
	}


	//find what power of 2, the number of processors is..
	//may cause errors if no of processors is not a exact power of 2
	q = 0;
	while( size > ( pow(2,q ) ) )
	{
	   q++;
	}

	// --------------- Stage 1: top-down -----------------------------
	double time = MPI_Wtime();
	for( level = 0; level <= q; level++ )
	{
	   //if is Reciever
	   if( isReciever( rank, size, level ) && level > 0 )
	   {
	      // receive the elements from the parent node
            MPI_Recv(a, n/(int) pow(2, level), MPI_DOUBLE, rank - size/(int) pow(2, level), 0, MPI_COMM_WORLD, &status);
	   }
	   //if is Active  ,
	   if( isActive( rank, size, level ) && level < q )
	   {
	      // send half of the elements to the right child node
          MPI_Send(&a[n/(int) pow(2, level + 1)], n/(int) pow(2, level + 1), MPI_DOUBLE, rank + size/(int) pow(2, level + 1), 0, MPI_COMM_WORLD);
	   }
	}
	double TComm = MPI_Wtime() - time;

	// --------------- Stage 2: sort the arrays -----------------------------

	time = MPI_Wtime();
	//sequential sort the scattered array
    merge_sort(n/size,a);
	double TComp = MPI_Wtime() - time;


	// --------------- Stage 3: bottom-up -----------------------------

	for( level = q; level >= 0 ; level-- )
	{
	   if( isActive( rank, size, level ) && level < q )
	   {
	      // receive the elements from the right child in the array b and then merge a with b
          time = MPI_Wtime();
		  MPI_Recv(b, n/(int) pow(2, level + 1), MPI_DOUBLE, rank + size/(int) pow(2, level + 1), 0, MPI_COMM_WORLD, &status);
		  TComm = TComm + (MPI_Wtime() - time);

		  time = MPI_Wtime();
          a = merge_array( n/(int) pow(2, level + 1), a,  n/(int) pow(2, level + 1), b);
		  TComp = TComp + (MPI_Wtime() - time);
	   }
	   if( isSender( rank, size, level ) && level > 0 )
	   {
	      // send the elements to the parent node.
		  time = MPI_Wtime();
          MPI_Send(a, n/(int) pow(2, level), MPI_DOUBLE, rank - size/(int) pow(2, level), 0, MPI_COMM_WORLD);
		  TComm = TComm + (MPI_Wtime() - time);
	   }
	}

	printf("Processor %d takes %fs for TComm + %fs for TComp, Total time = %fs\n", rank, TComm, TComp, TComm + TComp);


	if( rank == 0 )
	{   
        // printf("\nOutput: \n");
	   for( x = 0; x < n; x++ )
	   {
	    //   printf( "%f ", a[x] );
		//printf("Execution Time with %d procs is %lf", size, overallTime);
	   }
	}
	
	MPI_Finalize();

}



// ------------------------------------------------------------
//
// these functions test if a node is active, sender of receiver
//


int isActive( int rank, int p, int l )
{
   if( (rank  % ( (int)(p/pow(2,l)) ) )== 0 ) return 1;
   return 0;
}

int isReciever( int rank, int p, int l )
{
   if( !isActive( rank, p, l ) ) return 0;
   int result = rank / ((int)(p/pow(2,l)) );
   if( result % 2 == 1) return 1;
   return 0;
}

int isSender( int rank, int p, int l )
{
   return isReciever( rank, p , l );
}



// ------------------------------------------------------------
//
// these functions deal with sorting and merging
//

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

}

void swap (double * a, double * b){

   double temp;

   temp=*a;*a=*b;*b=temp;

}
