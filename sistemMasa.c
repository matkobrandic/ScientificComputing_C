#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "f2c.h"
#include "fblaswr.h"
#include "clapack.h"

void printm(integer n, doublereal* A){
    integer i,j;
    for(i=0;i<n;i++){
	    for(j=0;j<n;j++)
            printf("%.4lf ",A[i+n*j]);
        printf("\n");
    }
    printf("\n");
}
void printv(integer n, doublereal* x){
    integer i;
    for(i=0;i<n;i++) 
	    printf("%.4lf ",x[i]);
    printf("\n\n");
}

int main(){
    integer i;
    integer n,N;
    n = 4;N = n*n;
    doublereal *A = calloc(N,sizeof(doublereal));
    doublereal *M = calloc(N,sizeof(doublereal));
    doublereal *Mpol = calloc(N,sizeof(doublereal));
    
    M[0] = 2.0;
    M[5] = 5.0;
    M[10] = 3.0;
    M[15] = 6.0;
    
    Mpol[0] = 1/sqrt(2.0);
    Mpol[5] = 1/sqrt(5.0);
    Mpol[10] = 1/sqrt(3);
    Mpol[15] = 1/sqrt(6.0);
    doublereal K[]={24.0,-9.0,-5.0,0.0, -9.0,22.0,-8.0,-5.0, -5.0,-8.0,25.0,-7.0, 0.0,-5.0,-7.0,18.0};
    printf("K:\n");
    printm(n,K);
    printf("M:\n");
    printm(n,M);
    
    //A = M^(-1/2)*K*M^(-1/2) = Mpol*K*Mpol
    doublereal ALPHA = 1.0, BETA = 0.0;
    char TRANSA = 'N', TRANSB = 'N';
    dgemm_(&TRANSA,&TRANSB,&n,&n,&n,&ALPHA,Mpol,&n,K,&n,&BETA,A,&n); //A=Mpol*K
    doublereal *temp = calloc(N,sizeof(doublereal));
    integer INCX = 1, INCY = 1;
    dcopy_(&N,A,&INCX,temp,&INCY);
    dgemm_(&TRANSA,&TRANSB,&n,&n,&n,&ALPHA,temp,&n,Mpol,&n,&BETA,A,&n);
    
    printf("A:\n");
    printm(n,A);
    
    dcopy_(&N,A,&INCX,temp,&INCY);
    char JOBZ = 'V', UPLO = 'U';
    integer LDA=n, LDW=3*n-1, INFO;
    doublereal *W = calloc(n,sizeof(doublereal));
    doublereal *WORK = calloc(LDW,sizeof(doublereal));
    dsyev_(&JOBZ,&UPLO,&n,temp,&LDA,W,WORK,&LDW,&INFO);
    
    printv(n,W);
    //lambda = diag(W), U = temp
    doublereal *X = calloc(N,sizeof(doublereal));
    dgemm_(&TRANSA,&TRANSB,&n,&n,&n,&ALPHA,Mpol,&n,temp,&n,&BETA,X,&n);
     
    printf("x1 = ["); 
    for(i=0;i<n;i++)
        printf(" %.4lf",X[i]);
    printf("]*e^(i*t*%.4lf)\n",sqrt(W[0]));
    
    printf("x2 = ["); 
    for(i=0;i<n;i++)
        printf(" %.4lf",X[4+i]);
    printf("]*e^(i*t*%.4lf)\n",sqrt(W[1]));
    
    printf("x3 = ["); 
    for(i=0;i<n;i++)
        printf(" %.4lf",X[8+i]);
    printf("]*e^(i*t*%.4lf)\n",sqrt(W[2]));
    
    printf("x4 = ["); 
    for(i=0;i<n;i++)
        printf(" %.4lf",X[12+i]);
    printf("]*e^(i*t*%.4lf)\n",sqrt(W[3]));
    printf("\n");
    
    return 0;
}

