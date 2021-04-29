#include <stdio.h>
#include <stdlib.h>
#include "f2c.h"
#include "clapack.h"
#include "fblaswr.h"
#include <math.h>


void pcg(integer n, doublereal A[], doublereal b[], doublereal x0[], doublereal tol){
    int k = 0, i;
    char TRANS = 'N', UPLO = 'N';
    integer INCX = 1, INCY = 1;
    doublereal k_eps = 1, alpha_k, beta_k, beta_k_1;
    doublereal ALPHA = -1.0, BETA = 1.0, ALPHA1 = 1.0, BETA1 = 0.0;

    doublereal *M = malloc(n*n*sizeof(double));
	    for(i=0;i<n;i++)
	    	M[i*n+i]=A[i*n+i];
		

    doublereal *r_k = malloc(n*sizeof(double));
    dcopy_(&n, b, &INCX, r_k,&INCY);
    dgemv_(&TRANS, &n ,&n ,&ALPHA, A, &n, x0, &INCX, &BETA, r_k, &INCY);

    double *p_k = malloc(n*sizeof(double));
	
    char DIAG='N', UPLOU='U',TRANST='T';
    dcopy_(&n, r_k, &INCX, p_k, &INCY);
    dtrsv_(&UPLOU, &TRANST, &DIAG, &n, M, &n, p_k, &INCX);	// D^-1*p_k = r_k

    double *d_k = malloc(n*sizeof(double));
    dcopy_(&n, p_k, &INCX, d_k, &INCY);

    double norm_b = dnrm2_(&n,b,&INCX);

    doublereal old = ddot_(&n, r_k, &INCX, p_k, &INCY), new;
    double *Ad = malloc(n*sizeof(double));
    integer m = 1;
    double *pom = malloc(n*sizeof(double));

	while(k_eps>tol)
	{
		//inicijaliziraj mi alpha_k=(r_k'*p_k)/(d_k'*A*d_k)
		dgemv_(&TRANS, &n, &n, &ALPHA1, A, &n, d_k, &INCX, &BETA1, Ad, &INCY);
		alpha_k = old/ddot_(&n, d_k, &INCX, Ad, &INCY);
		
		// x_k = x_k + alpha_k*d_k
		daxpy_(&n, &alpha_k, d_k, &INCX, x0, &INCY);

		// r_k = r_k - alpha_k*A*d_k
		alpha_k = -alpha_k;
		daxpy_(&n, &alpha_k, Ad, &INCX, r_k, &INCY);
		
		// D^-1*p_k = r_k
		dcopy_(&n, r_k, &INCX, p_k, &INCY);
		dtrsv_(&UPLOU, &TRANST, &DIAG, &n, M, &n, p_k, &INCX);

		new = ddot_(&n, r_k, &INCX, p_k, &INCY);

		// beta_k=(r_k_1'*p_k_1)/(r_k'*p_k)
		beta_k=new/old;

		//inicijaliziraj mi d_k=p_k+beta_k*d_k
		dlacpy_(&UPLO, &n, &m, p_k, &n, pom, &n);
		daxpy_(&n, &beta_k, d_k, &INCX, pom, &INCY);
		dlacpy_(&UPLO, &n, &m, pom, &n, d_k, &n);

		k_eps=dnrm2_(&n, r_k, &INCX)/norm_b;
		old=new;
		k++;
	}
    printf("aproksimacija rješenja:\n");
    for(i=0; i<n; ++i){
        printf("%lf\n", x0[i]);
    }
    printf("\n");
    printf("Number of iterations needed for achieving the default accuracy: %d\n", k);
    }

int main (void){
    integer n = 100;
    doublereal *A, tol=1e-8, *x_0, *x_1, *b;
    int i, j;
    A = malloc (n*n*sizeof(double));
    x_0 = malloc(n*sizeof(double));
    x_1 = malloc(n*sizeof(double));
    b = malloc(n*sizeof(double));

	FILE *f;
	f=fopen("stieltjes_matr.txt","r");
	for (i=0;i<n*n;i++){
        fscanf(f,"%lf",A+i);
    }
	fclose(f);

    for(i=0;i<n;i++){
    	x_0[i]=0;
        x_1[i]=1;
    }

	char TRANS = 'N';
	doublereal alpha = 1.0;
	doublereal beta = 0.0;
	integer incx = 1;
	integer incy = 1;

	dgemv_( &TRANS, &n, &n, &alpha, A, &n, x_1, &incx, &beta, b, &incy );

    pcg(n, A, b, x_0, tol);

return 0;
}
