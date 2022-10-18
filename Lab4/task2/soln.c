
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int ** alloc_matrix(int n, int m);
void init_matrix(int n, int m, int ** a);
int ** prod_matrix(int n, int l, int m, int ** a, int ** b);
int ** trans_matrix(int n, int m, int ** a);


int main(int argc, char ** argv){
      int size, rank, tag=1, i,j,  n=10, **a, **b, **c, **b1, **c1, **tmp;

	MPI_Status stat;
	MPI_Datatype columntype, subtype, rowtocolumntype, subtwotype;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Type_vector(n, 1, n, MPI_INT, &columntype);
	MPI_Type_create_resized(columntype, 0, sizeof(MPI_INT), &subtype);
	MPI_Type_commit(&subtype);

	MPI_Type_vector(n, 1, n/size, MPI_INT, &rowtocolumntype);
	MPI_Type_create_resized(rowtocolumntype, 0, sizeof(MPI_INT), &subtwotype);
	MPI_Type_commit(&subtwotype);	

	a = alloc_matrix(n,n);
	b = alloc_matrix(n,n);
	c = alloc_matrix(n,n);

	b1=alloc_matrix(n, n/size);

	if (rank == 0) {
		init_matrix(n,n,a);
		init_matrix(n,n,b);
	}

	MPI_Scatter(b[0], n/size, subtype, b1[0], n/size, subtwotype, 0, MPI_COMM_WORLD);
	MPI_Bcast(a[0], n*n, MPI_INT, 0, MPI_COMM_WORLD);	

	c1 = prod_matrix(n, n, n/size, a, b1);

	MPI_Gather(c1[0], n/size, subtwotype, c[0], n/size, subtype, 0, MPI_COMM_WORLD);
	
	MPI_Type_free(&subtype);
        MPI_Type_free(&subtwotype);
        MPI_Finalize();

	free(a);
	free(b1);
	free(c1);
	free(b);

	if(rank == 1){
                for(i = 0; i < n; i++){
                        for(j = 0; j < n; j++)
                                printf("%5d ", c[i][j]);
                        printf("\n");
                }

        }
        free(c);
}


int ** alloc_matrix(int n, int m){

	int i, j, **a, *aa;

	aa=(int *) calloc(n*m, sizeof(int));
	a=(int **) calloc(n, sizeof(int*));

	for(i=0;i<n;i++)a[i]=aa+i*m;

	return a;
}


void init_matrix(int n, int m, int ** a){

	int i, j;

	for(i=0;i<n;i++)for(j=0;j<m;j++)a[i][j]= rand()%100;

	return a;
}


int ** prod_matrix(int n, int l, int m, int ** a, int ** b){

	int i,j,k,** c;

	c=alloc_matrix(n,m);

	for(i=0;i<n;i++)for(j=0;j<m;j++){

		c[i][j]=0;

		for(k=0;k<l;k++)c[i][j]=c[i][j]+a[i][k]*b[k][j];

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