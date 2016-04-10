#include "mathHelpers.h"


void MeanVar( double* restrict messung,int anz_mess, double*  restrict berechnet )
{
    berechnet[0] = 0.0;
    for( int i=0; i<anz_mess; i++ )
    {
        berechnet[0] += messung[i];
        berechnet[1] += messung[i]*messung[i];
    }

    berechnet[0] /=(double)anz_mess;
    berechnet[1] /= (double)anz_mess;
    berechnet[1] -= berechnet[0]*berechnet[0];
}

void fisherYates( int* restrict a, int N )
{
   for(int i=0; i<N; i++)
       a[i]=i;

   for(int i=N-1; i>=2; --i)
   {
       int j = (int)( MY_RND*(i-2) ) +1;
       int tmp = a[i];
       a[i] = a[j];
       a[j] = tmp;
   }
}

void printMatrixColWise( double* matrix, int N, int P )
{
    for( int n=0; n<N; n++ )
    {
        for( int p=0; p<P; p++)
           PRINT("%d  %+.3e\t", INDEX(n,p,N) , matrix[INDEX(n,p,N)]);
        PRINT("\n");
    }
}

void printMatrixRowWise( double* matrix, int N, int P )
{
    for( int n=0; n<N; n++ )
    {
        for( int p=0; p<P; p++ )
           PRINT("%zd  %+.3e\t", INDEX_ROW(n,p,N) , matrix[INDEX_ROW(n,p,N)]);
        PRINT("\n");
    }
}

void printVector( double* vector, int N )
{
    for( int i=0; i<N; i++ )
        PRINT( "%+.4e\n",vector[i]);
}

double sum( double* restrict a, int n )
{
    double tmp = 0.0;
    while( n-- != 0 )
    {
        const double atmp = *a;        
        tmp += atmp;
        a++;
    }
    return tmp;    
}

double squaresum( double* restrict a, int n )
{
    double tmp = 0.0;
    while( n-- != 0 )
    {
        const double atmp = *a;
        a++;
        tmp += atmp * atmp;
    }
    return tmp;    
}

double abssum( double* restrict a, int n )
{
    double tmp = 0.0;
    while( n-- != 0 )
    {
        const double atmp = *a;
        a++;
        tmp += fabs(atmp);
    }
    return tmp;    
}







