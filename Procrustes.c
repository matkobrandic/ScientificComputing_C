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
    integer m = 2, n = 4;
    doublereal A[] = {1.2,2.9, 5.2,6.8, 2.1,4.3, 6.1,8.1};
    doublereal B[] = {1.0,3.0,5.0,7.0,2.0,4.0,6.0,8.0};
    
    printf("\nA: \n");
    printm(m,n,A);
    
    printf("\nB: \n");
    printm(m,n,B);
    
    //C = A^T*B 
    doublereal *C = calloc(m*m,sizeof(doublereal));
    char TRANSA = 'T', TRANSB = 'N';
    doublereal ALPHA = 1.0, BETA = 0.0;
    integer LDA = n, LDB = n, LDC = m;
    dgemm_(&TRANSA, &TRANSB, &m, &m, &n, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
    
    printf("\nC: \n");
    printm(m,m,C);
    
    char JOBU = 'A', JOBVT = 'A';
    integer LDU = m, LDVT = m, LWORK = 5*m, INFO;
    LDA = m;
    doublereal *S = calloc(m,sizeof(doublereal)); //sing.vrijed. u padajucem poretku
    doublereal *U = calloc(m*m,sizeof(doublereal)); 
    doublereal *VT = calloc(m*m,sizeof(doublereal)); 
    doublereal *WORK = calloc(LWORK,sizeof(doublereal)); //pomocno polje 
    dgesvd_(&JOBU, &JOBVT, &m, &m, C, &LDA, S, U, &LDU, VT, &LDVT, WORK, &LWORK, &INFO);

    //Q = U*VT
    doublereal *Q = calloc(m*m,sizeof(doublereal)); 
    TRANSA = 'N';
    LDB = m;
    dgemm_(&TRANSA, &TRANSB, &m, &m, &m, &ALPHA, U, &LDA, VT, &LDB, &BETA, Q, &LDC);
    
    printf("\nQ: \n");
    printm(m,m,Q);
    
    doublereal min_norm;
    doublereal *D = calloc(m*n,sizeof(doublereal)); //D=AQ-B
    
    LDA = n;
    LDB = m;
    LDC = n;
    
    dgemm_(&TRANSA, &TRANSB, &n, &m, &m, &ALPHA, A, &LDA, Q, &LDB, &BETA, D, &LDC);

    for (i=0;i<8;i++)
        D[i] = D[i]-B[i];
        
    char NORM = 'F';
    LDA = m;
    min_norm = dlange_(&NORM, &m, &n, D, &LDA, WORK);
    printf("Minimal value is: %lf.\n\n",min_norm);
    return 0;
}
