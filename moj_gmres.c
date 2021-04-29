#include <stdio.h>
#include <stdlib.h>
#include "f2c.h"
#include "fblaswr.h"
#include "clapack.h"
#include <math.h>
#include "GMRES.c"
#include <complex.h>

char TRANS='N';
integer INC=1,n=100;
double *A;

int matvec (doublereal* ALPHA, doublereal* x, doublereal* BETA, doublereal* y){
	dgemv_(&TRANS,&n,&n,ALPHA,A,&n,x,&INC,BETA,y,&INC);
}


int psolve (doublereal* x, doublereal* b){
	dlacpy_(&TRANS,&n,&INC,b,&n,x,&n);
}

int main(){
	int i,j;
	double *g = (double*)malloc(n*sizeof(double));
	for(i=0;i<n;i++)
		g[i]=sqrt((n-i)*(n-i)-(n-i-1)*(n-i-1));
	integer IDIST = 2,m = n*n;
	integer ISEED[] = {432,13,134,69};
	double *V = (double*)malloc(n*n*sizeof(double));
	dlarnv_(&IDIST,ISEED,&m,V);
	double *tau = (double*)malloc(n*sizeof(double));
	double *WORK = (double*)malloc(n*sizeof(double));
	integer INFO;
	dgeqrf_(&n,&n,V,&n,tau,WORK,&n,&INFO);
	dorgqr_(&n,&n,&n,V,&n,tau,WORK,&n,&INFO);
	double *b = (double*)calloc(n,sizeof(double));
	for (i=0;i<n;i++)
		for(j=0;j<n;j++)
			b[i] = b[i] + g[j]*V[n*j+i];
	integer INCX = 1;	
	printf("Norm of b: \t%lf\n",dnrm2_(&n,b,&INCX));

	double *B = (double*)malloc(n*n*sizeof(double));	
	for (i=0;i<n*n;i++){
		if (i<n)
			B[i] = b[i];
		else
			B[i] = V[i-n];
	}
	double *D = (double*)calloc(n*n,sizeof(double));
	for (i=0;i<n;i++)
		for (j=0;j<n;j++){
				D[n*j+i] = 0;
				if (i == j+1) 
					D[n*j+i] = 1;
		}
	D[n*(n-1)] = 1;

	double *C = (double*)malloc(n*n*sizeof(double));
	char TRANSA = 'N',TRANSB = 'N';
	doublereal ALPHA = 1.0,BETA = 0.0;
	dgemm_(&TRANSA,&TRANSB,&n,&n,&n,&ALPHA,B,&n,D,&n,&BETA,C,&n);
	integer *IPIV = malloc(n*sizeof(integer));
	dgetrf_(&n,&n,B,&n,IPIV,&INFO);
	dgetri_(&n,B,&n,IPIV,WORK,&n,&INFO);

	A = (double*)malloc(n*n*sizeof(double));
	dgemm_(&TRANSA,&TRANSB,&n,&n,&n,&ALPHA,C,&n,B,&n,&BETA,A,&n);

	double *x = (double*)calloc(n,sizeof(double));
	integer LDH = n+1,restrt = n,LDW = n,iter = n;
	
	WORK = (double*)malloc(LDW*(n+4)*sizeof(double));
	double *h = (double*)malloc(LDH*(n+2)*sizeof(double));
	double resid = 1e-5;
    
    integer k = 3*n;	
	double *A1 = (double*)malloc(n*n*sizeof(double));
	dcopy_(&n,A,&n,A1,&n);
	double *wr = (double*)malloc(n*sizeof(double));
	double *wi = (double*)malloc(n*sizeof(double));
	double *vl = (double*)malloc(n*sizeof(double));
	double *vr = (double*)malloc(n*sizeof(double));
	double *work1 = (double*)malloc(k*sizeof(double));
	
	dgeev_(&TRANSA,&TRANSA,&n,A1,&n,wr,wi,vl,&n,vr,&n,work1,&k,&INFO);
	
	for(i=0;i<n;++i){
	    printf("%lf + %lfi\n", creal(cpow(wr[i]+wi[i]*I,100)),cimag(cpow(wr[i]+wi[i]*I,100)) );
	}
	printf("\n");
	

    gmres_(&n,b,x,&restrt,WORK,&LDW,h,&LDH,&iter,&resid,matvec,psolve,&INFO);

	return 0;
}


