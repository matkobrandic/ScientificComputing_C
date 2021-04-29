#include <stdio.h>
#include <stdlib.h>
#include "f2c.h"
#include "fblaswr.h"
#include "clapack.h"
#include "math.h"


void printm(doublereal *A,integer n){
	int i, j;
	printf("\n\n");	
	for(i = 0; i < n; i++){
		for(j = 0; j < n; j++)
			printf(" %f ",A[i+j*n]);
		printf("\n");
	}
	printf("\n\n");	
}

int main(){
	integer n = 21;
	integer N = n*n;
	integer m = 1;	
	int i, j;
	doublereal x[n];
	doublereal y[n];
	doublereal A[N];
	doublereal b[n];

	for(i = 0; i < n; i++){
		x[i] = -5+0.5*i;
		y[i] = -5+0.5*i;
	}
	for(i = 0; i < n; i++)
		for(j = 0; j < n; j++)
			A[i+j*n] = (x[i]*x[i]*y[j]-x[i]*x[i]-y[j]*y[j]+175)/250;

	b[0] = 10;
	for(i = 1; i < n; i++)
		b [i] = 1;

	integer jpvt[n];
	doublereal TAU[n];
	integer lwork = 3*n+1;
	doublereal work[lwork];
	integer info;
	dgeqp3_(&n,&n,A,&n,jpvt,TAU,work,&lwork,&info);
	//calculating rank
	integer r;
	for(i=0;i<n;i++)
		if(abs(A[i+i*n]) <= 21e-16){
			r = i;
			break;
		}	
	printf("Value of rank: %d\n",(int)r);	
	//first r rows of A is matrix R
	doublereal *R = calloc(r*n,sizeof(doublereal));
	for(i=0;i<r;i++)
		for(j=i;j<n;j++)
		    R[i+j*r] = A[i+j*n];
	//generating matrix Q
	dorgqr_( &n, &n, &n, A, &n, TAU, work, &lwork, &info );
	doublereal Q[N];
	char uplo;
    dlacpy_( &uplo, &n, &n, A, &n, Q, &n );

	//calculating Q^T*b
	char trans = 't';
	doublereal alpha = 1.0;
	doublereal beta = 0.0;
	doublereal z[n];
	dgemv_( &trans, &n, &n, &alpha, Q, &n, b, &m, &beta, z, &m );
	
	//taking only first 2 rows
	doublereal w[r];
	for(i=0;i<r;i++)
		w[i] = z[i];
	
	//LQ factorization of matrix R
	doublereal TAU1[r];
	integer lwork1 = r*n;
	doublereal work1[lwork1];
	dgelqf_( &r, &n, R, &r, TAU1, work1, &lwork1, &info );

	char side = 'l';
	uplo = 'l';
	trans = 'n';
	char diag = 'n';
	dtrsm_( &side, &uplo, &trans, &diag, &r, &m, &alpha, R, &r, w, &r );
	dorglq_( &r, &n, &r, R, &r, TAU1, work1, &lwork1, &info );
		    
	doublereal temp[n];
	trans = 'T';
	dgemv_( &trans, &r, &n, &alpha, R, &r, w, &m, &beta, temp, &m );

	doublereal solution[n];	
	for(i=0;i<n;++i)
		for(j=0;j<n;++j)
			if((int)jpvt[j] == i+1)
				solution[i] = temp[j];
	for(i=0;i<n;i++){
		if(solution[i]<0.0)
			printf("%f\n", solution[i]);
		else
			printf(" %f\n",solution[i]);
	}
	return 0;
}		
			
	
