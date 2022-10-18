#include "mpi.h" 
#include <stdio.h>

// function declarations

// function defs

int main(int argc, char * argv []) {
 
	int numtasks, rank, dest, source, rc, count, tag=1;  
	char inmsg, outmsg; 
	MPI_Status Stat ;

	// Init MPI_COMM_WORLD
	MPI_Init (&argc,&argv); 

	// Init size and rank
	MPI_Comm_size (MPI_COMM_WORLD, &numtasks); 
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);

	if (rank == 0) {
	  dest = source = 1;outmsg=’x’;
	  rc = MPI_Send (&outmsg, 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
	  rc = MPI_Recv (&inmsg, 1, MPI_CHAR, source, tag, MPI_COMM_WORLD, &Stat);
	} 
	else if (rank == 1) {
	  dest = source = 0;outmsg=’y’;
	  rc = MPI_Recv (&inmsg, 1, MPI_CHAR, source, tag, MPI_COMM_WORLD, &Stat);
	  rc = MPI_Send (&outmsg, 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
 	}

	rc = MPI_Get_count (&Stat, MPI_CHAR, &count);
	printf("Task %d: Received %d char(s) from task %d with tag %d \n", rank, count, Stat.MPI_SOURCE,Stat.MPI_TAG); 
	MPI_Finalize ();
}// MPI program to ping-pong between Processor 0 and Processor 1

#include "mpi.h" 
#include <stdio.h>

int main(int argc, char * argv []) {
 
	int numtasks, rank, dest, source, rc, count, tag=1;  
	char inmsg, outmsg; 
	MPI_Status Stat ;

	MPI_Init (&argc,&argv); 
	MPI_Comm_size (MPI_COMM_WORLD, &numtasks); 
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);

	if (rank == 0) {
	  dest = source = 1;outmsg=’x’;
	  rc = MPI_Send (&outmsg, 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
	  rc = MPI_Recv (&inmsg, 1, MPI_CHAR, source, tag, MPI_COMM_WORLD, &Stat);
	} 
	else if (rank == 1) {
	  dest = source = 0;outmsg=’y’;
	  rc = MPI_Recv (&inmsg, 1, MPI_CHAR, source, tag, MPI_COMM_WORLD, &Stat);
	  rc = MPI_Send (&outmsg, 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
 	}

	rc = MPI_Get_count (&Stat, MPI_CHAR, &count);
	printf("Task %d: Received %d char(s) from task %d with tag %d \n", rank, count, Stat.MPI_SOURCE,Stat.MPI_TAG); 
	MPI_Finalize ();
}
