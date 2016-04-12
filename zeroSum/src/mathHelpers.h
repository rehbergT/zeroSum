#ifndef MATHHELPERS_H
#define MATHHELPERS_H

 #define R_PACKAGE

#ifdef R_PACKAGE

    #include <R.h>
    #include <Rdefines.h>
    #include <R_ext/Utils.h>
    #include <R_ext/Rdynload.h>

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

#define TRUE  1
#define FALSE 0

void MeanVar( double messung[],int anz_mess,  double berechnet[]);
void fisherYates(int* a, int N);

void printMatrixColWise(double* matrix, int N, int P);
void printMatrixRowWise(double* matrix, int N, int P);
void printVector(double* vector, int N);

double squaresum(double *a, int n);
double abssum(double *a, int n);
double sum(double *a, int n);


#endif /* MATHHELPERS_H */


