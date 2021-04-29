#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "f2c.h"
#include "fblaswr.h"
#include "clapack.h"

void cut(integer marker, doublereal *Z, integer m, doublereal *Dpol){
	integer i,j;
	if(marker == 1){
		printf("V1: ");
		for(i=0;i<m;i++)
			if (Z[m+i] >= 0) printf("%ld ",i+1);
		printf("\nV2: ");
		for(i=0;i<m;i++)
			if (Z[m+i] < 0) printf("%ld ",i+1);
		printf("\n");
	}
	if (marker == 2){
		printf("V1: ");
		for(i=0;i<m;i++)
			if (Dpol[i+m*i] * Z[m+i] >= 0) printf("%ld ",i+1);
		printf("\nV2: ");
		for(i=0;i<m;i++)
			if (Dpol[i*m+i] * Z[m+i] < 0) printf("%ld ",i+1);
		printf("\n");
	}
}

void printm(integer n, doublereal* A){ //print matrix
	integer i,j;
	for(i=0;i<n;i++){
		for(j=0;j<n;j++)
        	printf("%.4lf ",A[i+n*j]);
		printf("\n");
	}
	printf("\n");
}

void printv(integer n, doublereal* x){ //print vector
	integer i;
	for(i=0;i<n;i++) 
		printf("%.4lf ",x[i]);
	printf("\n\n");
}

int main(){
	integer i,j;
	integer n = 7;
	integer N = n*n;
	doublereal *D = calloc(n*n,sizeof(doublereal));
	D[0] = 9;   //2+4+3
	D[8] = 10;  //2+7+1
	D[16] = 9;  //3+3+2+1
	D[24] = 14; //3+4+7
	D[32] = 11; //1+7+3
	D[40] = 14; //7+5+2
	D[48] = 9;  //5+3+1

//printm(n,D);

//fill in only the lower triangle then copy the upper part
doublereal *W = calloc(n*n,sizeof(doublereal)); 
	W[1] = W[19] = 2;
	W[2] = W[17] = W[34] = 3;
	W[11] = W[20] = 1;
	W[10] = W[33] = 7;
	W[41] = 5;
	W[3] = 4;

for(i=0;i<n;i++)
	for(j=0;j<i;j++)
		W[i*n+j]=W[j*n+i];
//printm(n,W);

//Laplace graph matrix = L = D - W
	doublereal *L = calloc(n*n,sizeof(doublereal));
	for(i=0;i<n;i++)
		for(j=0;j<n;j++)
			L[i*n+j] = D[i*n+j] - W[i*n+j];
	printf("Laplace graph matrix:\n");
	printm(n,L);

//normalized Laplace graph matrix = LN
	doublereal *LN = calloc(n*n,sizeof(doublereal));
	//A = M^(-1/2) * K * M^(-1/2) = Mpol * K * Mpol
	//LN = D^(-1/2) * L * D^(-1/2)
	doublereal *Dpol = calloc(n*n,sizeof(doublereal));
	for(i=0;i<n;i++)
		for(j=0;j<n;j++)
			if(D[i*n+j] != 0) Dpol[i*n+j] = 1/sqrt(D[i*n+j]);
	//printm(n,Dpol);
	doublereal ALPHA = 1.0,BETA = 0.0;
	char TRANSA = 'N',TRANSB = 'N';
	dgemm_(&TRANSA,&TRANSB,&n,&n,&n,&ALPHA,Dpol,&n,L,&n,&BETA,LN,&n); //LN = Dpol*L
	doublereal *temp = calloc(N,sizeof(doublereal));
	integer INCX = 1,INCY = 1;
	dcopy_(&N,LN,&INCX,temp,&INCY);
	dgemm_(&TRANSA,&TRANSB,&n,&n,&n,&ALPHA,temp,&n,Dpol,&n,&BETA,LN,&n); //LN = (Dpol*L)*Dpol
	printf("Normalized Laplace graph matrix:\n");
	printm(n,LN);

//dsyevx
	char JOBZ = 'V';
	char RANGE = 'A'; //?
	char UPLO = 'U';
	doublereal VL,VU;
	integer IL,IU;
	char abs = 's';
	double ABSTOL = 2*dlamch_(&abs);
	integer M = n;
	doublereal *Z = malloc(n*M*sizeof(doublereal));
	integer LDZ = n, LWORK = 8*n;
	doublereal *WORK = malloc(LWORK*sizeof(doublereal));
	integer *IWORK = malloc(5*n*sizeof(integer));
	integer *IFAIL = malloc(n*sizeof(integer));
	integer INFO;
	dsyevx_(&JOBZ, &RANGE, &UPLO, &n, L, &n, &VL, &VU, &IL, &IU, &ABSTOL, &M, W, Z, &LDZ, WORK, &LWORK, IWORK, IFAIL, &INFO); //sv.vrijed i sv.vektroi od L

	/*if((int)INFO==0) printf("Success!!!\n");
	printf("There are %ld eigenvalues.\n", M);
	printf("Eigenvectors:\n");
	printm(M,Z);
	printf("Eigenvalues:\n");
	for(i=0; i<M; i++) printf("%lf ",W[i]);
	printf("\n");*/

	if((int)INFO==0){
		printf("Second least eigenvalue of L: %lf.\n",W[1]);
		printf("Fielder's vector:\n");
		for(i=0; i<n; i++) printf("%lf ",Z[n+i]);
		printf("\n\n");
		cut(1,Z,M,Dpol);
	}
	
	dsyevx_(&JOBZ, &RANGE, &UPLO, &n, LN, &n, &VL, &VU, &IL, &IU, &ABSTOL, &M, W, Z, &LDZ, WORK, &LWORK, IWORK, IFAIL, &INFO); //sv.vrijed i sv.vektroi od LN
	if((int)INFO==0){
		printf("\nSecond least eigenvalue of LN: %lf.\n",W[1]);
		printf("Normalized Field's vector:\n");
		for(i=0; i<n; i++) printf("%lf ",Z[n+i]);
		printf("\n\n");
		cut(2,Z,M,Dpol);
	}
	return 0;
}
