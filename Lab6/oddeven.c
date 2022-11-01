	

    // include headers
    # include <stdio.h>
    # include <math.h>
    # include <stdlib.h>
    # include <mpi.h>
         
    //MPI methods
    int MPI_Exchange( int n, double * array, int rank1, int rank2, MPI_Comm comm );
    int MPI_Sort_oddeven( int n, double * array, int root, MPI_Comm comm );
    int MPI_Is_sorted(int n, double * array, int * answer, int root, MPI_Comm comm);
     
    //all in previous labs
    double * merge( int n, double * array, int m, double * b );
    void merge_sort( int n, double * array );
    void swap ( double * array, double * b );
     
    // function definitions
    int main( int argc, char ** argv ) {
      //setup
      int size, rank, result, i, *answer;
      int n = 10000000;
      double m = 10.0;
      double *array, processorTime;
     
      MPI_Status status;
      MPI_Init( &argc, &argv );
      MPI_Comm_rank( MPI_COMM_WORLD, &rank );
      MPI_Comm_size( MPI_COMM_WORLD, &size );
     
      //allocate space for an array of doubles, size n
      array = ( double * ) calloc( n, sizeof( double ) );
       
      //fills array with random values on root proc
      if( rank == 0 )
      {
        //get random values for array & output for testing
        srand( ( ( unsigned ) time( NULL ) + rank ) );
        for( i = 0; i < n; i++ )
        {
          array[ i ] = ( ( double ) rand( ) / RAND_MAX ) * m;
          // printf( "Initial: %f\n", array[ i ] );
        }
      }
     
      //get start time for each processor
      processorTime = MPI_Wtime( );
     
      //MPI_Sort does all the heavy work
      // call and time evaluate MPI_Sort_direct
      double time = MPI_Wtime();
      double overallTime;
      result = MPI_Sort_oddeven( n, array, 0, MPI_COMM_WORLD );
      if( result != MPI_SUCCESS )
      {
        return result;
      }
      time = MPI_Wtime() - time;
      MPI_Reduce(&time, &overallTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      if(rank == 0) {
          //for(int i = 0; i < n; i++) printf("%lf ", array[i]);
          printf("\nExecution Time with %d procs is %lf\n", size, overallTime);
      }
     
      //get end time for each processor
      processorTime = MPI_Wtime( ) - processorTime;
      //TODO COMMENT OUT
      printf( "Processor %d takes %lf sec\n", rank, processorTime );
     
      // output ordered list for testing
      if( rank == 0 )
      {
        for( i = 0; i < n; i++ )
        {
          //TODO COMMENT OUT
			    //printf( "Output : %f\n", array[ i ] );
        }
      }
      MPI_Finalize( );
    }
     
    int MPI_Sort_oddeven( int n, double * array, int root, MPI_Comm comm ) {
      
      // get rank and size of comm
      int rank, size;
      MPI_Comm_rank( MPI_COMM_WORLD, &rank );
      MPI_Comm_size( MPI_COMM_WORLD, &size );
      
      //allocate space for numElements/numProcessors amount of doubles
      double * localArray = ( double * ) calloc( n / size, sizeof( double ) );
         
      //scatter a to local_a
      int rc = MPI_Scatter(array, n / size, MPI_DOUBLE, localArray, n / size, MPI_DOUBLE, root, comm);
      if(rc != MPI_SUCCESS) return rc;
       
      //sort local_a using mergeSort
      merge_sort(n/size, localArray);
       
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

     
      //gather local_a
        MPI_Gather( localArray, n/size, MPI_DOUBLE, array, n/size, MPI_DOUBLE, 0, MPI_COMM_WORLD );

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
     
    //notes
    void merge_sort( int n, double * a ) {
      double * c;
      int i;
     
      if ( n <= 1 )
      {
        return;
      }
      if( n == 2 )
      {
        if( a[ 0 ] > a[ 1 ] )
        {
          swap( &a[ 0 ], &a[ 1 ] );
        }
        return;
      }
     
      merge_sort( n / 2, a );
      merge_sort( n - n / 2, a + n / 2 );
      c = merge( n / 2, a, n - n / 2, a + n / 2);
      for( i = 0; i < n; i++ )
      {
        a[ i ] = c[ i ];
      }
    }
     
    //notes
    void swap ( double * a, double * b ) {
       double temp;
       temp = *a;
       *a = *b;
       *b = temp;
    }
