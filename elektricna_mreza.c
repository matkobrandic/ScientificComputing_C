#include <stdio.h>
#include <stdlib.h>
#include "f2c.h"
#include "fblaswr.h"
#include "clapack.h"
#include <math.h>
#include "GMRES.c"

char TRANS='N';
integer INC=1,n=6;
double A[]={11.0,-20.0,0.0,0.0,0.0,-2.0,-5.0,41.0,-3.0,0.0,-3.0,0.0,0.0,-15.0,7.0,-1.0,0.0,0.0,0.0,0.0,-4.0,2.0,-10.0,0.0,0.0,-6.0,0.0,-1.0,28.0,-15.0,-1.0,0.0,0.0,0.0,-15.0,47.0};
int matvec (doublereal* alpha, doublereal* x, doublereal* beta, doublereal* y){
	dgemv_(&TRANS,&n,&n,alpha,A,&n,x,&INC,beta,y,&INC);
}

int psolve (doublereal* x, doublereal* b){
	dlacpy_(&TRANS,&n,&INC,b,&n,x,&n);
}

void sor_solver(integer n, doublereal A[], doublereal omega, doublereal epsilon, doublereal b[], doublereal x0[]){
	doublereal *y = malloc(n*sizeof(double));
	int k = 0,i,j;
	doublereal BETA = 1.0,ALPHA = -1.0,pom;
	integer INCX = 1,INCY = 1;
	char TRANS = 'N';
	dcopy_(&n,b,&INCX,y,&INCY);
	double k_eps = 1.0;
	while(k_eps>epsilon){
		k++;
		for(i=0;i<n;++i){
			x0[i] = (1-omega)*x0[i];
			pom = b[i];
			for(j=0;j<i;++j){
				pom = pom - A[i+j*n]*x0[j];
			}
			for(j=i+1; j<n; ++j){
				pom = pom - A[i+j*n]*x0[j];
			}
			x0[i] = x0[i] + (pom*omega)/A[i+i*n];
		}
		dcopy_(&n,b,&INCX,y,&INCX);
		dgemv_(&TRANS,&n,&n,&ALPHA,A,&n,x0,&INCX,&BETA,y,&INCY);
		k_eps = (dnrm2_(&n,y,&INCX))/dnrm2_(&n,b,&INCX);
	}
	printf("Number of iterations needed for reaching defined approximation using SOR method: %d\n", k);
	printf("Solution approximation:\n");
	for(i=0;i<n;++i){
		printf(" %lf ", x0[i]);
	}
	printf("\n\n");
}

void GS(integer n, doublereal A[], doublereal omega, doublereal epsilon, doublereal b[], doublereal x0[]){
	doublereal *y = malloc(n*sizeof(double));
	int k=0,i,j;
	doublereal BETA = 1.0,ALPHA = -1.0,pom;
	integer INCX = 1,INCY = 1;
	char TRANS = 'N';
	dcopy_(&n,b,&INCX,y,&INCY);
	double k_eps = 1.0;
	while(k_eps>epsilon){
		k++;
		for(i=0;i<n;++i){
			x0[i] = (1-omega)*x0[i];
			pom = b[i];
			for(j=0; j<i; ++j){
				pom = pom - A[i+j*n]*x0[j];
			}
			for(j=i+1; j<n; ++j){
				pom = pom - A[i+j*n]*x0[j];
			}
			x0[i] = x0[i] + (pom*omega)/A[i+i*n];
		}
		dcopy_(&n, b, &INCX, y, &INCX);
		dgemv_(&TRANS, &n, &n, &ALPHA, A, &n, x0, &INCX, &BETA, y, &INCY);
		k_eps = (dnrm2_(&n, y, &INCX))/dnrm2_(&n, b, &INCX);
	}
	printf("Number of iterations needed for reaching defined approximation using GS method: %d\n", k);
	printf("Solution approximation:\n");
	for(i=0;i<n;++i){
		printf(" %lf ", x0[i]);
	}
	printf("\n\n"); 
}

int main(){
	int i,j;
	integer INFO;
	double x_gs[] = {0.0,0.0,0.0,0.0,0.0,0.0}, x_sor[] = {0.0,0.0,0.0,0.0,0.0,0.0}, x_gmres[] = {0.0,0.0,0.0,0.0,0.0,0.0};
	double b[] = {500.0,0.0,0.0,0.0,0.0,0.0};
	integer ldh = n+1,restrt = n,ldw = n,iter = n;
	double *WORK = (double*)malloc(ldw*(n+4)*sizeof(double));
	double *h = (double*)malloc(ldh*(n+2)*sizeof(double));
	double *resid = (double*)malloc(n*n*sizeof(double));

	doublereal omega = 1.0,epsilon = 1e-8;
	GS(n,A,omega,epsilon,b,x_gs);
	omega = 1.35;
	sor_solver(n,A,omega,epsilon,b,x_sor);
	gmres_(&n,b,x_gmres,&restrt,WORK,&ldw,h,&ldh,&iter,resid,matvec,psolve,&INFO);
	
	printf("\nNumber of iterations needed for reaching defined approximation using GMRES method: %d\n", (int)iter);
	printf("Solution approximation:\n");
	for(i=0; i<n; ++i){
		printf(" %lf ", x_gmres[i]);
	}
	printf("\n\n"); 
	
	return 0;
}

