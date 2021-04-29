#include <stdio.h>
#include <stdlib.h>
#include "f2c.h"
#include "fblaswr.h"
#include "clapack.h"
#include "math.h"
 
 
int main(){
    integer n = 21;
    integer N = n*n;
    integer m = 1;
    int i,j;
    doublereal x[n],y[n],A[N],b[n];

    for(i=0;i<n-1;++i){
        x[i] = -5+0.5*i;
        y[i] = -5+0.5*i;
    }
    for(i=0;i<n;++i)
        for(j=0;j<n;++j){
            A[i+j*n] = (x[i]*x[i]*y[j] - x[i]*x[i] - y[j]*y[j] + 175)/250;
        }
    b[0] = 10;
    for(i=1;i<n;++i)
        b[i] = 1;
//QR
    integer jpvt[n];
    doublereal TAU[n];
    integer lwork = 3*n+1;
    doublereal work[lwork];
    integer info;
    dgeqp3_(&n,&n,A,&n,jpvt,TAU,work,&lwork,&info);
//rank
    integer r;
    for(i=0;i<n;++i){
        if(abs(A[i+i*n])<=21e-16){
            r = i;
            break;
        }   
    }
    printf("Rank: %d\n",(int)r);   
//First r rows of A is R
    doublereal *R=calloc(r*n,sizeof(doublereal));
    for(i=0;i<r;++i)
        for(j=i;j<n;++j)
            R[i+j*r] = A[i+j*n];
//generating Q
    dorgqr_(&n,&n,&n,A,&n,TAU,work,&lwork,&info);
    doublereal Q[N];
    char uplo;
    dlacpy_(&uplo,&n,&n,A,&n,Q,&n);
//computing Q^T*b
    char trans = 't';
    doublereal alpha = 1.0;
    doublereal beta = 0.0;
    doublereal z[n];
    dgemv_(&trans,&n,&n,&alpha,Q,&n,b,&m,&beta,z,&m);
//using only first two rows
    doublereal w[r];
    for(i=0;i<r;++i)
        w[i] = z[i];
//LQ factorization of R
    doublereal TAU1[r];
    integer lwork1 = r*n;
    doublereal work1[lwork1];
    dgelqf_(&r,&n,R,&r,TAU1,work1,&lwork1,&info);
//x = L^-1*w
//L is lower triangle of R, solution is saved in w
    char side = 'l';
    uplo = 'l';
    trans = 'n';
    char diag = 'n';
    dtrsm_(&side,&uplo,&trans,&diag,&r,&m,&alpha,R,&r,w,&r);
//saving Q from LQ factorization of R in R -> matrix Z from the assignment
    dorglq_(&r,&n,&r,R,&r,TAU1,work1,&lwork1,&info);
//R^T*w 
    doublereal temp[n];
    trans = 'T';
    dgemv_(&trans,&r,&n,&alpha,R,&r,w,&m,&beta,temp,&m);
//solution permutation
    doublereal solution[n];
    for(i=0;i<n;++i)
        for(j=0;j<n;++j)
            if(jpvt[j] == i+1)
                solution[i] = temp[j];
    printf("PRINTING SOLUTION:\n");
    for(i=0;i<n;++i)
        printf("%f\n", solution[ i ] );      
    return 0;
} 
