#include "mpi.h" 
#include <stdlib.h>
#include <stdio.h>

// function declarations
int testRPS(char move1, char move2);
char* moves = "RPS";

// function definitions

int main(int argc, char * argv []) {

	// Init MPI_COMM_WORLD
	MPI_Init (&argc,&argv); 

	int size, rank;
	MPI_Comm_size (MPI_COMM_WORLD, &size); 
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);

	// generate a random number bwt 0, 1, 2
	unsigned int iseed = (unsigned int) time(NULL);
	srand(iseed*rank);       // seed the generator with a different seed.
	int move = rand()%3, otherMove;    // generate a random number between 0-2.

	// ping pong move
	int tag1 = 1, tag2 = 2, winner;
	MPI_Status status;


	if (rank == 0) {
	  MPI_Send (&move, 1, MPI_INT, 1, tag1, MPI_COMM_WORLD);
	  MPI_Recv (&otherMove, 1, MPI_INT, 1, tag2, MPI_COMM_WORLD, &status);

	  winner = testRPS(moves[move], moves[otherMove]);
	} 
	else if (rank == 1) {
	  MPI_Recv (&otherMove, 1, MPI_INT, 0, tag1, MPI_COMM_WORLD, &status);
	  MPI_Send (&move, 1, MPI_INT, 0, tag2, MPI_COMM_WORLD);

	  winner = testRPS(moves[otherMove], moves[move]);
 	}

	printf("RPS winner is %d on processor %d\n", winner, rank);

	MPI_Finalize();

	return 0;
}

int testRPS(char move1, char move2) {
	if (move1 == move2) return -1;
	if ((move1 == 'R' && move2 == 'S') || (move1 == 'S' && move2 == 'P') || (move1 == 'P' && move2 == 'R')) {
		return 0;
	} else {
		return 1;
	}
}
