#include <stdio.h>
#include <stdlib.h>
#include "f2c.h"
#include "fblaswr.h"
#include "clapack.h"
#include <math.h>


int sgn(doublereal val) {
    return (0.0 < val) - (val < 0.0);
}


int jacobi_sd(double *A,integer n, double epsilon){
	double S=0;
	int i,j;
	for(i=0;i<n;i++)
		for(j=0;j<n;j++)
			if(i!=j)
				S=S+A[n*j+i]*A[n*j+i];	
	S=sqrt(S);			
			
	char NORM='F';
	double *WORK;
	WORK = (double*)malloc(n*sizeof(double));
	double norma_F = dlange_(&NORM,&n,&n,A,&n,WORK); 

	int p,q,k,sign;
	double tau,t,c,s,app,apq,aqq,pom;
	while(S > epsilon*norma_F){
		for(p=0;p<n-1;p++){
			for(q=p+1;q<n;q++){

				if(A[q*n+p]!=0){
					tau = (A[q*n+q]-A[p*n+p])/(2*A[q*n+p]);
					t = sgn(tau)/(fabs(tau)+sqrt(1+pow(tau,2)));	
					c = 1/(sqrt(1+pow(t,2)));
					s = t*c;			
				}
				else{
				    c=1;
				    s=0;
				}
				apq = A[q*n+p];
				app = A[p*n+p];
				aqq = A[q*n+q];
				app = app-t*apq;
				aqq = aqq+t*apq;
			
				for(k=0;k<n;k++){
					pom = A[p*n+k];
					A[p*n+k] = c*pom-s*A[q*n+k];
					A[q*n+k] = s*pom+c*A[q*n+k];
					A[k*n+p] = A[p*n+k];
					A[k*n+q] = A[q*n+k];
				}
				A[q*n+p] = 0;
				A[p*n+q] = 0;
				A[p*n+p] = app;
				A[q*n+q] = aqq;		
			}
		}
		S = 0;
		for(i=0;i<n;i++)
		    for(j=0;j<n;j++)
			    if(i!=j)if(i!=j)
				S=S+A[n*j+i]*A[n*j+i];	
		S=sqrt(S);
			
		norma_F=dlange_(&NORM,&n,&n,A,&n,WORK);
	}	

	for(i=0;i<n;i++)
		printf("%f\n",A[i*n+i]);

	return 0;
}

int main(){

//4x4 matrix

	integer n = 4;
	double epsilon = 4*(1e-16);

	integer IDIST = 2, m = n*n;
	integer ISEED[] = {70,211,32,469};
	double *U = (double*)malloc(n*n*sizeof(double));
	dlarnv_(&IDIST,ISEED,&m,U);
	double *tau = (double*)malloc(n*sizeof(double));
	double *WORK = (double*)malloc(n*sizeof(double));
	integer INFO; 
	dgeqrf_(&n,&n,U,&n,tau,WORK,&n,&INFO);
	dorgqr_(&n,&n,&n,U,&n,tau,WORK,&n,&INFO);
	
	double D[] = {-10.0,0.0,0.0,0.0,0.0,-5.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.2};

	double *C = (double*)malloc(n*n*sizeof(double));
	char TRANSA = 'N',TRANSB = 'N';
	doublereal ALPHA = 1.0, BETA = 0.0;
	dgemm_(&TRANSA,&TRANSB,&n,&n,&n,&ALPHA,U,&n,D,&n,&BETA,C,&n);
	double *A = (double*)malloc(n*n*sizeof(double));
	TRANSB = 'T';
	dgemm_(&TRANSA,&TRANSB,&n,&n,&n,&ALPHA,C,&n,U,&n,&BETA,A,&n);

printf("A = U*D*U^T 4x4 solution:\n");

	jacobi_sd(A,n,epsilon);

//RIS MATRIX

	int i,j;
	n = 10;
	epsilon = 1e-15;
	double *R = (double*)malloc(n*n*sizeof(double));
	for(i=0;i<n;i++)
		for(j=0;j<n;j++)
			R[j*n+i] = 1/(2*(8-i-j+1.5));

printf("Ris matrix 10x10 solution:\n");

	jacobi_sd(R,n,epsilon);
	return 0;
}
