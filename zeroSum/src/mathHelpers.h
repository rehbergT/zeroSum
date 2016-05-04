#ifndef MATHHELPERS_H
#define MATHHELPERS_H

 #define R_PACKAGE

#ifdef R_PACKAGE

    #include <R.h>
    #include <Rdefines.h>
    #include <R_ext/Utils.h>
    #include <R_ext/Rdynload.h>
    
    // #include<Rmath.h>
    // #include<R_ext/BLAS.h>
    #define PRINT Rprintf
    #define MY_RND unif_rand()

#else  
    #include <stdio.h>
    #define PRINT printf
    #define MY_RND gsl_rng_uniform(rngGSLPointer)
    
#endif

#include <math.h>
#include <stdlib.h>

#define INDEX(i,j,N) ( (i) + ( (j) * (N) ) )
#define INDEX_ROW(i,j,M) ( (j) + ( (i) * (M) ) )

// http://www.mathcs.emory.edu/~cheung/Courses/255/Syllabus/1-C-intro/bit-array.html
#define SetBit(A,k)     ( A[(k/32)] |= (1 << (k%32)) )
#define ClearBit(A,k)   ( A[(k/32)] &= ~(1 << (k%32)) )
#define TestBit(A,k)    ( A[(k/32)] & (1 << (k%32)) )

#define TRUE  1
#define FALSE 0

void MeanVar( double messung[],int anz_mess,  double berechnet[]);
void fisherYates(int* restrict a, const int N);

void printMatrixColWise(double* matrix, int N, int P);
void printMatrixRowWise(double* matrix, int N, int P);
void printVector(double* vector, int N);

double scalarProdSquaresum(double* restrict w, double* restrict a, const int n );
double scalarProdSum(double* restrict w, double* restrict a, const int n );

double squaresum(double* restrict a, const int n);
double abssum(double* restrict a, const int n);
double sum(double* restrict a, const int n);

double absSumDiffMult( double* restrict a,
                       double* restrict b,
                       double* restrict c,
                       const int n  );


#endif /* MATHHELPERS_H */


