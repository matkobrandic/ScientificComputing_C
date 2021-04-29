#include <stdio.h>
#include <stdlib.h>
#include "f2c.h"
#include "clapack.h"
#include "fblaswr.h"
#include <math.h>


void cg(integer n, doublereal A[], doublereal b[], doublereal x0[], doublereal tol){
    int k = 0, i;
    char TRANS = 'N';
    integer INCX = 1, INCY = 1;
    doublereal *y, k_eps = 1, alpha_k, beta_k, beta_k_1, *d_k, *dd_k, *r_k, *r_k_1;
    doublereal ALPHA = -1.0, BETA = 1.0, ALPHA1 = 1.0, BETA1 = 0.0;
    
    // dk = rk = b-Ax0
    d_k = malloc(n*sizeof(double));
    r_k = malloc(n*sizeof(double));
    r_k_1 = malloc(n*sizeof(double));
    dd_k = malloc(n*sizeof(double));
    y = malloc(n*sizeof(double));
    
    dcopy_(&n, b, &INCX, d_k, &INCX);
    dcopy_(&n, b, &INCX, r_k, &INCX);

    dgemv_(&TRANS, &n, &n, &ALPHA, A, &n, x0, &INCX, &BETA, d_k, &INCY);
    dcopy_(&n, d_k, &INCX, r_k, &INCX);

    double norm_b = dnrm2_(&n,b,&INCX);;
	while(k_eps>tol){

		// alpha_k = (r_k'*r_k)/(d_k'*A*d_k)
		dgemv_(&TRANS, &n, &n, &ALPHA1, A, &n, d_k, &INCX, &BETA1, dd_k, &INCY);
		alpha_k=ddot_(&n, r_k, &INCX, r_k, &INCY)/ddot_(&n, d_k, &INCX, dd_k, &INCY);

		// x_k = x_k + alpha_k*d_k
		daxpy_(&n, &alpha_k, d_k, &INCX, x0, &INCY);


		// r_k_1 = r_k - alpha_k*A*d_k
		alpha_k=-alpha_k;
		dcopy_(&n, r_k, &INCX, r_k_1, &INCX);
		daxpy_(&n, &alpha_k, dd_k, &INCX, r_k_1, &INCY);

		// beta_k = (r_k_1'*r_k_1)/(r_k'*r_k)
		beta_k=ddot_(&n, r_k_1, &INCX, r_k_1, &INCY)/ddot_(&n, r_k, &INCX, r_k, &INCY);

		// d_k = r_k_1+beta_k*d_k
		dcopy_(&n, r_k_1, &INCX, dd_k, &INCX);
		daxpy_(&n, &beta_k, d_k, &INCX, dd_k, &INCY);
		dcopy_(&n, dd_k, &INCX, d_k, &INCX);
	
		// r_k = r_k_1
		dcopy_(&n, r_k_1, &INCX, r_k, &INCX);	

		//dcopy_(&n, b, &INCX, y, &INCX);
		//dgemv_(&TRANS, &n, &n, &ALPHA, A, &n, x0, &INCX, &BETA, y, &INCY);	//b-A*x0
		k_eps=dnrm2_(&n,r_k_1,&INCX)/norm_b;
		
		k=k+1;
		printf("Step number %d.\n Accuracy achieved in %d. step: %lf \n", k, k, k_eps);
	}

    printf("Solution approximation:\n");
    for(i=0; i<n; ++i){
        printf(" %lf \n", x0[i]);
    }
    printf("\n");
    printf("Number of iterations needed for achieving the default accuracy: %d\n", k);
}

int main (void){
    integer n = 100;
    doublereal *A, tol=1e-8, *x_0, *x_1, *b, br=0.0;
    int i, j;
    A = malloc (n*n*sizeof(double));
    x_0 = malloc(n*sizeof(double));
    x_1 = malloc(n*sizeof(double));
    b = malloc(n*sizeof(double));
    integer N = n*n, idist = 2, iseed[] = {17,1589,144,7};
    dlarnv_( &idist, iseed, &N, A );
    doublereal *Lambda;
	Lambda = malloc(n*n*sizeof(double));
	for(j=0;j<n;j++){	
		for(i=0;i<n;i++){
			if(i==j){
				br++;
				Lambda[i+j*n] = br*br;
			}
			else 
				Lambda[i]=0;
        }
	}
    Lambda[0] = 1;

    doublereal *tau;
    tau = malloc(n*sizeof(double));	
    doublereal *work;
    work = malloc(n*sizeof(double));	
    integer info=0;
    dgeqrf_(&n,&n,A,&n,tau,work,&n,&info);
    char side='L',trans='N';	
    
    dormqr_(&side,&trans,&n,&n,&n,A,&n,tau,Lambda,&n,work,&n,&info);
    side='R',trans='T';
    dormqr_(&side,&trans,&n,&n,&n,A,&n,tau,Lambda,&n,work,&n,&info);
    for(i=0;i<n;i++){
    	x_0[i]=0;
    	x_1[i]=1;
    }
	trans = 'N';
	doublereal alpha = 1.0;
	doublereal beta = 0.0;
	integer incx = 1;
	integer incy = 1;
	dgemv_( &trans, &n, &n, &alpha, Lambda, &n, x_1, &incx, &beta, b, &incy );	//ovdje računam vektor b

cg(n, Lambda, b, x_0, tol);

return 0;
}
