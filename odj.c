#include <stdio.h>
#include <stdlib.h>
#include "f2c.h"
#include "fblaswr.h"
#include "clapack.h"
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
	}

    printf("Solution approximation:\n");
    for(i=0; i<n; ++i){
        printf(" %lf \n", x0[i]);
    }
    printf("\n");
    printf("Number of iterations needed for achieving the default accuracy: %d\n", k);
}

void sor_solver(integer n, doublereal A[], doublereal omega, doublereal epsilon, doublereal b[], doublereal x0[])
{
 
doublereal *y=malloc(n*sizeof(double));

    int k=0, i, j; doublereal BETA=1.0, ALPHA=-1.0, pom; integer INCX=1, INCY=1; char TRANS='N';
    dcopy_(&n, b, &INCX, y, &INCY);
    double k_eps=1.0;

    while(k_eps>epsilon){
        k++;
        for(i=0; i<n; ++i){
            x0[i]=(1-omega)*x0[i];
            pom = b[i];
            for(j=0; j<i; ++j){
                pom = pom - A[i+j*n]*x0[j];
            }
            for(j=i+1; j<n; ++j){
                pom = pom - A[i+j*n]*x0[j];
            }
            x0[i] = x0[i]+(pom*omega)/A[i+i*n];
        }
        dcopy_(&n, b, &INCX, y, &INCX);
        dgemv_(&TRANS, &n, &n, &ALPHA, A, &n, x0, &INCX, &BETA, y, &INCY);
        k_eps = (dnrm2_(&n, y, &INCX))/dnrm2_(&n, b, &INCX);
    }
    printf("Solution approximation:\n");
    for(i=0; i<n; ++i){
        printf(" %lf\n", x0[i]);
    }
    printf("\n");
    printf("Number of iterations needed for reaching defined approximation using SOR method: %d\n", k);
}

void GS(integer n, doublereal A[], doublereal omega, doublereal epsilon, doublereal b[], doublereal x0[]){
    doublereal *y = malloc(n*sizeof(double));
    int k=0, i, j; doublereal BETA=1.0, ALPHA=-1.0, pom;
    integer INCX=1, INCY=1; char TRANS='N';
    dcopy_(&n, b, &INCX, y, &INCY);
    double k_eps=1.0;

    while(k_eps>epsilon){
        k++;
        for(i=0; i<n; ++i){
            x0[i] = (1-omega)*x0[i];
            pom = b[i];
            for(j=0; j<i; ++j){
                pom = pom - A[i+j*n]*x0[j];
            }
            for(j=i+1; j<n; ++j){
                pom = pom - A[i+j*n]*x0[j];
            }
            x0[i]=x0[i]+(pom*omega)/A[i+i*n];
        }
        dcopy_(&n, b, &INCX, y, &INCX);
        dgemv_(&TRANS, &n, &n, &ALPHA, A, &n, x0, &INCX, &BETA, y, &INCY);
        k_eps=(dnrm2_(&n, y, &INCX))/dnrm2_(&n, b, &INCX);
    }
    printf("Solution approximation:\n");
    for(i=0; i<n; ++i){
        printf(" %lf\n", x0[i]);
    }
    printf("\n");
    printf("Number of iterations needed for reaching defined approximation using GS method: %d\n", k);
}

int main(){
	integer n = 99;
	int i,j;
	double epsilon = 1e-8;

	double *A = (double*)malloc(n*n*sizeof(double));
	for (i=0;i<n;i++){
		for (j=0;j<n;j++){
				A[n*j+i] = 0;
				if (i == j+1) 
					A[n*j+i] = -1;
				if (j == i+1)
					A[n*j+i] = -1;
				if(i == j)
					A[n*j+i] = 1.9999; 			
		}
    }
	
	double *b = (double*)malloc(n*sizeof(double));
	for(i=0;i<n;i++)
		b[i] = 0.0002*sin(0.01*(i+1));
	b[n-1] = 0.0002*sin(0.99)+cos(1);
	
	double *y = (double*)malloc(n*sizeof(double));
	for(i=0;i<n;i++)
		y[i] = (i+1)*0.01*cos((i+1)*0.01);

	double *gs_x = (double*)calloc(n,sizeof(double));
	doublereal omega = 1.0;
	GS(n, A, omega, epsilon, b, gs_x);

	double *gs_difference = (double*)malloc(n*sizeof(double));
	for(i=0;i<n;i++)
		gs_difference[i] = gs_x[i]-y[i];
	printf("GS -> Exact solution and approximation difference:\n");
	for(i=0;i<n;++i){
        printf(" %lf\n", gs_difference[i]);
	}
	printf("\n");

	double *sor_x = (double*)calloc(n,sizeof(double));
	omega = 1.95;
	sor_solver(n, A, omega, epsilon, b, sor_x);
	double *sor_difference = (double*)malloc(n*sizeof(double));
	for(i=0;i<n;i++)
		sor_difference[i] = sor_x[i]-y[i];
	printf("SOR -> Exact solution and approximation difference:\n");
	for(i=0;i<n;++i){
    	printf(" %lf\n", sor_difference[i]);
	}
	printf("\n");

	double *cg_x = (double*)calloc(n,sizeof(double));
	cg(n, A, b, cg_x, epsilon);
	double *cg_difference = (double*)malloc(n*sizeof(double));
	for(i=0;i<n;i++)
		cg_difference[i] = cg_x[i]-y[i];
	printf("CG -> Exact solution and approximation difference:\n");
	for(i=0;i<n;++i){
    	printf(" %lf\n", cg_difference[i]);
	}
	printf("\n");

	char UPLO = 'U';
	integer INFO;
	dpotrf_(&UPLO, &n, A, &n, &INFO);
    printf("\nIs A pozitivno definitna?\n");
	if(INFO==0){
	printf("Yes.\n");
	}
	else{
	printf("No.\n");
	}
	return 0;
}
