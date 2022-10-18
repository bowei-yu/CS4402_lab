
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int ** alloc_matrix(int n, int m);
void init_matrix(int n, int m, int ** a);
int ** prod_matrix(int n, int l, int m, int ** a, int ** b);
int ** trans_matrix(int n, int m, int ** a);


int main(int argc, char ** argv){
      int size, rank, tag=1, i,j,  n=10, **a, **b, **c, **a1, **c1;

	MPI_Status stat;
	MPI_Datatype columntype;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	a=alloc_matrix(n,n);
	b=alloc_matrix(n,n);
	c=alloc_matrix(n,n);

	a1=alloc_matrix(n/size,n);

	if (rank == 0) {

		init_matrix(n,n,a);
		init_matrix(n,n,b);

	}
	//Scatter A and BCast B to all
	MPI_Scatter(a[0], n*n/size, MPI_INT, a1[0], n*n/size, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(b[0], n*n, MPI_INT, 0, MPI_COMM_WORLD);
	//Compute product c1 = a1*b
	c1 = prod_matrix(n/size, n, n, a1, b);	

	//Gather c1 back onto root (as c)
	MPI_Gather(c1[0], n*n/size, MPI_INT, c[0], n*n/size, MPI_INT, 0, MPI_COMM_WORLD);
	//Write elements of c (as root)
	if(rank == 0){
		for(i = 0; i < n; i++){
			for(j = 0; j < n; j++)
				printf("%5d ", c[i][j]);
			printf("\n");
		}
	}
	
	MPI_Finalize();
}


int ** alloc_matrix(int n, int m){

	int i, j, **a, *aa;

	aa=(int *) calloc(n*m, sizeof(int));
	a=(int **) calloc(n, sizeof(int*));

	for(i=0;i<n;i++)a[i]=aa+i*m;

	for(i=0;i<n;i++)for(j=0;j<m;j++)a[i][j]=0;

	return a;
}


void init_matrix(int n, int m, int ** a){

	int i, j;

	for(i=0;i<n;i++)for(j=0;j<m;j++)a[i][j]= rand()%100;

	//return a;
}


int ** prod_matrix(int n, int l, int m, int ** a, int ** b){

	int i,j,k,** c;

	c = alloc_matrix(n,m);

	for(i = 0; i < n; i++)
		for(j = 0; j < m; j++){
			c[i][j]=0;

			for(k=0;k<l;k++)
				c[i][j] = c[i][j] + a[i][k] * b[k][j];
	}

	return c;

}


int ** trans_matrix(int n, int m, int ** a){

        int i,j;
	int ** b;

	b=alloc_matrix(m,n);

	for(j=0;j<m;j++)for(i=0;i<n;i++)b[j][i]=a[i][j];

	return b;

}