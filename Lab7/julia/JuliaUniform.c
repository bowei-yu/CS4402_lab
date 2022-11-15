#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include "mpe.h"
#include "mpe_graphics.h"
#include "complex.h"

Comp func(Comp x, Comp c){
	Comp res, temp;
	temp = CompProd(x,x);
	c = CompAdd(c, temp);
	return c;
}

int main( int argc, char** argv )
{
	// MPI size, rank variables
	int size, rank;

	// MPI Wtime variables
	double my_time, max_time;

	// MPE Variables
	MPE_XGraph graph;
	MPE_Color my_color, *colors;

	int numColors=100;

	// Window variables
	int width = 800;

	// Constants for Julia Sets
	// defines target area
	const int NUMITER = 200;
	const double XMIN = -1;
	const double XMAX =  1;
	const double YMIN = -1;
	const double YMAX =  1;
	const double R    = 10;


	// the complex number to generate Julia set for
	// for each complex number, there is julia set
	const double C_RE = 0.37; const double C_IM = 0.1;

	const double STEP = (XMAX-XMIN)/width;

	// Misc variables
	int i,j;

	char ckey;

	// Complex number variables
	Comp z, c;

	// Start of program
	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPE_Open_graphics( &graph, MPI_COMM_WORLD, NULL, -1, -1, width, width, 0 );

	// create a colour for each processor
	colors = (MPE_Color *) malloc((numColors) * sizeof(MPE_Color));

	MPE_Make_color_array (graph, numColors, colors);


	// contruct the complex number c
	c.re = C_RE; c.im = C_IM;

	my_time = MPI_Wtime();

	// schedule the parallel loops onto processors
	for(i=rank; i<=width; i++) for(j=0; j<width; j++)
	{
		int k =0;

		// transform i, j into complex number

		// contruct the first complex number
		z.re = XMIN + i*STEP;
		z.im = YMIN + j*STEP;

		// find its orbit by generating NUMITER iterations, if it goes to infinity, then break
		for (k=0; k < NUMITER; k++) {
			z = func(z,c);
			if (CompAbs(z) > R)
				break;
		}

		if (k>NUMITER-1) { // finite
			MPE_Draw_point(graph, i,j, MPE_YELLOW);
			MPE_Update(graph);
		}
		else { // infinity
			MPE_Draw_point(graph, i,j, colors[k%numColors]);
			MPE_Update(graph);
		}
	}

	my_time = MPI_Wtime()- my_time;
	printf("%lf Time taken for process %d is %lf\n",MPI_Wtime(),rank,my_time);
	fflush(stdout);

	if ( rank == 0 ) {
		printf( "Hit any key then return to continue  " );
		fflush(stdout);
		scanf( "%c", &ckey ); printf( "\n" );
	}
	MPE_Close_graphics( &graph );
	MPI_Finalize();

	return 0;
}
