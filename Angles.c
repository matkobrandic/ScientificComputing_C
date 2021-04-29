#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "f2c.h"
#include "fblaswr.h"
#include "clapack.h"

void printm(integer m, integer n, doublereal* A){
    integer i,j;
    for(i=0;i<n;i++){
	    for(j=0;j<m;j++)
            printf("%.4lf ",A[i+n*j]);
        printf("\n");
    }
    printf("\n");
}

int main(){
    integer i;
    integer m = 4, n = 2;
    doublereal A[] = {1,0,0,0,1,1,1,1};
    doublereal B[] = {1,-1,1,-1,0,1,0,1};
    //OR factorization
    integer LDA = m, LWORK = n*n, INFO;
    doublereal *TAUA = malloc(n*sizeof(doublereal));
    doublereal *TAUB = malloc(n*sizeof(doublereal));
    doublereal *WORK = malloc(LWORK*sizeof(doublereal));
    
    dgeqrf_(&m, &n, A, &LDA, TAUA, WORK, &LWORK, &INFO);
    dgeqrf_(&m, &n, B, &LDA, TAUB, WORK, &LWORK, &INFO);
    
    dorgqr_( &m, &n, &n, A, &m, TAUA, WORK, &LWORK, &INFO );
    dorgqr_( &m, &n, &n, B, &m, TAUB, WORK, &LWORK, &INFO );

    doublereal *C = malloc(n*n*sizeof(doublereal));
    char TRANSA = 'T', TRANSB = 'N';
    doublereal ALPHA = 1.0, BETA = 0.0;
    LDA = m;
    integer LDB = m, LDC = n;
    dgemm_(&TRANSA, &TRANSB, &n, &n, &m, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC); 
    
    char JOBU = 'A', JOBVT = 'A';
    integer LDU = n, LDVT = n;
    LDA = n;
    LWORK = 5*n;
    doublereal *S = calloc(n,sizeof(doublereal));
    doublereal *U = calloc(n*n,sizeof(doublereal)); 
    doublereal *VT = calloc(n*n,sizeof(doublereal)); 
    doublereal *WORK1 = calloc(LWORK,sizeof(doublereal)); //pomocno polje 
    dgesvd_(&JOBU, &JOBVT, &n, &n, C, &LDA, S, U, &LDU, VT, &LDVT, WORK1, &LWORK, &INFO);

    printf("\nAngles cos: ");
    for(i=0;i<n;i++)
        printf(" %lf ",S[i]);
    printf("\n\n");
    
    doublereal *X = malloc(m*n*sizeof(doublereal));
    doublereal *Y = malloc(m*n*sizeof(doublereal));
    
    dgemm_( &TRANSB, &TRANSB, &m, &n, &n, &ALPHA, A, &m, U, &n, &BETA, X, &m );
    dgemm_( &TRANSB, &TRANSA, &m, &n, &n, &ALPHA, B, &m, VT, &n, &BETA, Y, &m );
    
    printf("X:\n");
    printm(n,m,X);
    
    printf("Y:\n");
    printm(n,m,Y);
    
    printf("Angle between the planes: %lf.\n\n",acos(S[1]));
    
    return 0;
}
