#include "mathHelpers.h"


void MeanVar(   double* restrict messung, 
                const int anz_mess, 
                double* restrict berechnet )
{
    berechnet[0] = sum( messung, anz_mess) / (double)anz_mess;
    berechnet[1] = squaresum( messung, anz_mess) / (double)anz_mess;
    berechnet[1] -= berechnet[0]*berechnet[0];
}

void fisherYates( int* restrict a, const int N )
{
   for( int i=0; i<N; ++i )
       a[i]=i;

   for( int i=N-1; i>=2; --i )
   {
       int j = (int)( MY_RND*(i-2) ) + 1;
       int tmp = a[i];
       a[i] = a[j];
       a[j] = tmp;
   }
}

void printMatrixColWise( double* matrix, int N, int P )
{
    for( int n=0; n<N; ++n )
    {
        for( int p=0; p<P; ++p )
           PRINT("%d  %+.3e\t", INDEX(n,p,N) , matrix[INDEX(n,p,N)]);
        PRINT("\n");
    }
}

void printMatrixRowWise( double* matrix, int N, int P )
{
    for( int n=0; n<N; ++n )
    {
        for( int p=0; p<P; ++p )
           PRINT("%zd  %+.3e\t", INDEX_ROW(n,p,N) , matrix[INDEX_ROW(n,p,N)]);
        PRINT("\n");
    }
}

void printVector( double* vector, int N )
{
    for( int i=0; i<N; i++ )
        PRINT( "i=%d  %+.4e\n", i, vector[i]);
}

double sum( double* restrict a, const int n )
{
    if(n < 8)
    {
        double sum = 0.0;
        for ( int i=0; i<n; ++i )
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
        for( ; i < n-(n%8); i += 8 )
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
        for ( int i=0; i<n; ++i )
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
        for ( int i=0; i<n; ++i )
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


double scalarProdSquaresum( double* restrict w, double* restrict a, const int n )
{
    if(n < 8)
    {
        double sum = 0.0;
        for ( int i=0; i<n; ++i )
            sum += w[i]*a[i]*a[i];
        
        return sum;
    }
    else 
    {
        double psum[8];
        psum[0] = w[0] * a[0] * a[0];
        psum[1] = w[1] * a[1] * a[1];
        psum[2] = w[2] * a[2] * a[2];
        psum[3] = w[3] * a[3] * a[3];
        psum[4] = w[4] * a[4] * a[4];
        psum[5] = w[5] * a[5] * a[5];
        psum[6] = w[6] * a[6] * a[6];
        psum[7] = w[7] * a[7] * a[7];

        int i=8;
        for( ; i<n-(n%8); i += 8 )
        {
            psum[0] += w[i + 0] * a[i + 0] * a[i + 0];
            psum[1] += w[i + 1] * a[i + 1] * a[i + 1];
            psum[2] += w[i + 2] * a[i + 2] * a[i + 2];
            psum[3] += w[i + 3] * a[i + 3] * a[i + 3];
            psum[4] += w[i + 4] * a[i + 4] * a[i + 4];
            psum[5] += w[i + 5] * a[i + 5] * a[i + 5];
            psum[6] += w[i + 6] * a[i + 6] * a[i + 6];
            psum[7] += w[i + 7] * a[i + 7] * a[i + 7];
        }
    
        double sum = 0.0;
        for( int j=0; j<8; ++j )
            sum += psum[j];

        for ( ; i<n; ++i)
            sum += w[i]*a[i]*a[i];     
        return sum;
    }
}

double scalarProdSum( double* restrict w, double* restrict a, const int n )
{
    if(n < 8)
    {
        double sum = 0.0;
        for ( int i=0; i<n; ++i )
            sum += w[i]*a[i];
        
        return sum;
    }
    else 
    {
        double psum[8];
        psum[0] = w[0] * a[0];
        psum[1] = w[1] * a[1];
        psum[2] = w[2] * a[2];
        psum[3] = w[3] * a[3];
        psum[4] = w[4] * a[4];
        psum[5] = w[5] * a[5];
        psum[6] = w[6] * a[6];
        psum[7] = w[7] * a[7];

        int i=8;
        for( ; i<n-(n%8); i += 8 )
        {
            psum[0] += w[i + 0] * a[i + 0];
            psum[1] += w[i + 1] * a[i + 1];
            psum[2] += w[i + 2] * a[i + 2];
            psum[3] += w[i + 3] * a[i + 3];
            psum[4] += w[i + 4] * a[i + 4];
            psum[5] += w[i + 5] * a[i + 5];
            psum[6] += w[i + 6] * a[i + 6];
            psum[7] += w[i + 7] * a[i + 7];
        }
    
        double sum = 0.0;
        for( int j=0; j<8; ++j )
            sum += psum[j];

        for ( ; i<n; ++i)
            sum += w[i]*a[i];     
        return sum;
    }
}


double absSumDiffMult( double* restrict a,
                       double* restrict b,
                       double* restrict c,
                       const int n  )
{
    if(n < 8)
    {
        double sum = 0.0;
        for ( int i=0; i<n; ++i )
            sum += ( a[i]-b[i] ) * c[i];
        
        return fabs(sum);
    }
    else 
    {
        double psum[8];
        psum[0] = (a[0] - b[0]) * c[0];
        psum[1] = (a[1] - b[1]) * c[1];
        psum[2] = (a[2] - b[2]) * c[2];
        psum[3] = (a[3] - b[3]) * c[3];
        psum[4] = (a[4] - b[4]) * c[4];
        psum[5] = (a[5] - b[5]) * c[5];
        psum[6] = (a[6] - b[6]) * c[6];
        psum[7] = (a[7] - b[7]) * c[7];

        int i=8;
        for( ; i<n-(n%8); i += 8 )
        {
            psum[0] += (a[i + 0] - b[i + 0]) * c[i + 0];
            psum[1] += (a[i + 1] - b[i + 1]) * c[i + 1];
            psum[2] += (a[i + 2] - b[i + 2]) * c[i + 2];
            psum[3] += (a[i + 3] - b[i + 3]) * c[i + 3];
            psum[4] += (a[i + 4] - b[i + 4]) * c[i + 4];
            psum[5] += (a[i + 5] - b[i + 5]) * c[i + 5];
            psum[6] += (a[i + 6] - b[i + 6]) * c[i + 6];
            psum[7] += (a[i + 7] - b[i + 7]) * c[i + 7];
        }
    
        double sum = 0.0;
        for( int j=0; j<8; ++j )
            sum += psum[j];

        for ( ; i<n; ++i)
            sum += (a[i] - b[i]) * c[i];
        return fabs(sum);
    }
  
    
}


