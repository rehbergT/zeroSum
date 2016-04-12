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

double sum( double* restrict a, const int n )
{
    if(n < 8)
    {
        double sum = 0.0;
        for ( int i=0; i<n; ++i)
            sum += a[i];
        
        return sum;
    }
    else 
    {
        double psum[8];
        psum[0] = a[0];
        psum[1] = a[1];
        psum[2] = a[2];
        psum[3] = a[3];
        psum[4] = a[4];
        psum[5] = a[5];
        psum[6] = a[6];
        psum[7] = a[7];

        int i=8;
        for( ; i<n-(n%8); i += 8 )
        {
            psum[0] += a[i + 0];
            psum[1] += a[i + 1];
            psum[2] += a[i + 2];
            psum[3] += a[i + 3];
            psum[4] += a[i + 4];
            psum[5] += a[i + 5];
            psum[6] += a[i + 6];
            psum[7] += a[i + 7];
        }
    
        double sum = 0.0;
        for( int j=0; j<8; ++j )
            sum += psum[j];

        for ( ; i<n; ++i)
            sum += a[i];        
        return sum;
    }
}



double squaresum( double* restrict a, const int n )
{
    if(n < 8)
    {
        double sum = 0.0;
        for ( int i=0; i<n; ++i)
            sum += a[i]*a[i];
        
        return sum;
    }
    else 
    {
        double psum[8];
        psum[0] = a[0] * a[0];
        psum[1] = a[1] * a[1];
        psum[2] = a[2] * a[2];
        psum[3] = a[3] * a[3];
        psum[4] = a[4] * a[4];
        psum[5] = a[5] * a[5];
        psum[6] = a[6] * a[6];
        psum[7] = a[7] * a[7];

        int i=8;
        for( ; i<n-(n%8); i += 8 )
        {
            psum[0] += a[i + 0] * a[i + 0];
            psum[1] += a[i + 1] * a[i + 1];
            psum[2] += a[i + 2] * a[i + 2];
            psum[3] += a[i + 3] * a[i + 3];
            psum[4] += a[i + 4] * a[i + 4];
            psum[5] += a[i + 5] * a[i + 5];
            psum[6] += a[i + 6] * a[i + 6];
            psum[7] += a[i + 7] * a[i + 7];
        }
    
        double sum = 0.0;
        for( int j=0; j<8; ++j )
            sum += psum[j];

        for ( ; i<n; ++i)
            sum += a[i]*a[i];        
        return sum;
    }
}

double abssum( double* restrict a, const int n )
{
    if(n < 8)
    {
        double sum = 0.0;
        for ( int i=0; i<n; ++i)
            sum += fabs(a[i]);
        
        return sum;
    }
    else 
    {
        double psum[8];
        psum[0] = fabs(a[0]);
        psum[1] = fabs(a[1]);
        psum[2] = fabs(a[2]);
        psum[3] = fabs(a[3]);
        psum[4] = fabs(a[4]);
        psum[5] = fabs(a[5]);
        psum[6] = fabs(a[6]);
        psum[7] = fabs(a[7]);

        int i=8;
        for( ; i<n-(n%8); i += 8 )
        {
            psum[0] += fabs(a[i + 0]);
            psum[1] += fabs(a[i + 1]);
            psum[2] += fabs(a[i + 2]);
            psum[3] += fabs(a[i + 3]);
            psum[4] += fabs(a[i + 4]);
            psum[5] += fabs(a[i + 5]);
            psum[6] += fabs(a[i + 6]);
            psum[7] += fabs(a[i + 7]);
        }
    
        double sum = 0.0;
        for( int j=0; j<8; ++j )
            sum += psum[j];

        for ( ; i<n; ++i)
            sum += fabs(a[i]);        
        return sum;
    }
}







