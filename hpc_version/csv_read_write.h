#ifndef CSV_READ_WRITE_H
#define CSV_READ_WRITE_H

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include "../zeroSum/src/regressions.h"

#define CSV_MAX_LINE_LENGTH 1e8
#define MAX_SEQUENCE 1e5
#define SEP ","

void readCsvAsMatrix(char* path, double* matrix, int N, int M, int mN )
{
    int n,m; // Matrix dimensions
    int i,j; // loop variables
    FILE * file = fopen(path, "r");

    if( file == NULL )
    {
        printf("Error: file pointer is null\n");
        exit(1);
    }

    // define a temp char array for storing a line of the csv file
    char* lineTemp = (char*)malloc( CSV_MAX_LINE_LENGTH * sizeof(char) );
    char* linePointer = NULL;

    // get dimensions of the matrix
    n=0;
    m=0;
    while( fgets( lineTemp, CSV_MAX_LINE_LENGTH, file ) )
    {
        ++n;
        if(n==1)
        {
            linePointer = strtok( lineTemp, SEP );

            while( linePointer != NULL  )
            {
                linePointer = strtok( NULL, SEP );
                ++m;
            }
        }
    }
    // reduce by one because of header line
    --n;
    // reduce by one because of row names
    --m;
    // reset file pointer
    rewind(file);
    if(n != N || m != M)
    {
        printf("Error: wrong dimensions of csv file\n");
        exit(1);
    }

    i=0;
    j=0;

    // first line are colnames -> read & drop it
    linePointer = fgets( lineTemp, CSV_MAX_LINE_LENGTH, file );

    while( fgets( lineTemp, CSV_MAX_LINE_LENGTH, file ) )
    {
        linePointer = strtok( lineTemp, SEP );
        for(j=0; j<m; ++j)
        {
            linePointer = strtok( NULL, SEP );
            matrix[INDEX(i,j,mN)] = strtod ( linePointer, NULL );
        }
        i++;
    }

    fclose(file);
    free(lineTemp);
}


void readCsvAsFusion( char* path, RegressionData* data )
{
    FILE * file = fopen(path, "r");

    if( file == NULL )
    {
        printf("Error: file pointer is null\n");
        exit(1);
    }

    // define a temp char array for storing a line of the csv file
    char* lineTemp = (char*)malloc( CSV_MAX_LINE_LENGTH * sizeof(char) );
    char* linePointer = fgets( lineTemp, CSV_MAX_LINE_LENGTH, file );

    int i,j;
    double x;

    while( fgets( lineTemp, CSV_MAX_LINE_LENGTH, file ) )
    {
        // two times since first element is the rowname
        linePointer = strtok( lineTemp, SEP );
        linePointer = strtok( NULL, SEP );
        i = (int)strtol( linePointer, NULL, 10 );
        linePointer = strtok( NULL, SEP );
        j = (int)strtol( linePointer, NULL, 10 );
        linePointer = strtok( NULL, SEP );
        x = strtod ( linePointer, NULL );
        // printf("i=%d j=%d x=%e\n",i,j,x);
        data->fusionKernel[j] = appendElement( data->fusionKernel[j], i, x);
    }

    fclose(file);
    free(lineTemp);
}

double* readCsvSave(char* path, int* N, int* M )
{
    int n,m; // Matrix dimensions
    int i,j; // loop variables
    FILE * file = fopen(path, "r");

    // define a temp char array for storing a line of the csv file
    char* lineTemp = (char*)malloc( CSV_MAX_LINE_LENGTH * sizeof(char) );
    char* linePointer = NULL;

    // get dimensions of the matrix
    n=0;
    m=0;
    while( fgets( lineTemp, CSV_MAX_LINE_LENGTH, file ) )
    {
        ++n;
        if(n==1)
        {
            linePointer = strtok( lineTemp, SEP );

            while( linePointer != NULL  )
            {
                linePointer = strtok( NULL, SEP );
                ++m;
            }
        }
    }

    rewind(file);
    *N = n;
    *M = m;
    i=0;
    j=0;

    double* matrix = (double*)malloc(n*m*sizeof(double));

    while( fgets( lineTemp, CSV_MAX_LINE_LENGTH, file ) )
    {
        for(j=0; j<m; ++j)
        {
            if(j==0)
                linePointer = strtok( lineTemp, SEP );
            else
                linePointer = strtok( NULL, SEP );
            matrix[INDEX(i,j,n)] = strtod ( linePointer, NULL );
        }
        i++;
    }


    fclose(file);
    free(lineTemp);
    return(matrix);
}


void readSaves( char* path, char* name, RegressionData* data)
{
    FILE* file;
    char filePath[MAX_PATH_LENGTH];
    int fileNumber=0;

    // find the smallest commen lambda in each
    double lambdaCommon=-2.0;

    while(TRUE)
    {
        snprintf( filePath, sizeof(filePath),
                  "%s%s_%d_stats.csv",
                  path, name, fileNumber  );

        file = fopen(filePath, "r");
        if( file==NULL ) break;
        fclose(file);

        PRINT("found: %s\n", filePath);

        int rows, cols;
        double* content = readCsvSave( filePath,  &rows, &cols );

        int min = getMinIndex(&content[INDEX(0,1,rows)], rows);
        PRINT("rows: %d cols: %d lambdaMin: %f\n", rows, cols, content[INDEX(min,1,rows)] );

        if(fileNumber==0)
            lambdaCommon = content[INDEX(min,1,rows)];
        else if( content[INDEX(min,1,rows)] > lambdaCommon )
            lambdaCommon = content[INDEX(min,1,rows)];

        fileNumber++;
        free(content);
    }

    if( lambdaCommon != -2.0 )
    {
        printf("lambdaCommon: %e", lambdaCommon);
        int index =-1;
        for(int i=0; i<data->lengthLambda; i++)
        {
            if( fabs(data->lambdaSeq[i]-lambdaCommon)< DBL_EPSILON*100 )
                index=i;
        }

        if(index==data->lengthLambda-1)
        {
            perror("Abborted! Everythings seems to be calculated\n");
            MPI_Abort( MPI_COMM_WORLD, 1);
        }

        data->lengthLambda = data->lengthLambda-index-1;
        memmove(data->lambdaSeq, &data->lambdaSeq[index+1], data->lengthLambda * sizeof(double) );
        PRINT("Found save files starting from lambda: %f\n", lambdaCommon);

    }
}

RegressionData* readRegressionData( int argc, char **argv )
{
    if( argc != 6 && argc != 7 )
    {
        PRINT("Error: only %d arguments\n", argc);
        exit(1);
    }
    RegressionData* data = NULL;

    char* lineTemp   =   (char*)malloc( CSV_MAX_LINE_LENGTH * sizeof(char) );
    double* sequence = (double*)malloc( MAX_SEQUENCE * sizeof(double));

    char* linePointer = NULL;

    FILE * file;
    file = fopen(argv[1], "r");

    if(file == NULL)
    {
        PRINT("Error: file pointer is null\n");
        exit(1);
    }

    int N=0, P=0, K=0, nc=0, type=0;

    // first line are colnames -> read & drop it
    linePointer = fgets( lineTemp, CSV_MAX_LINE_LENGTH, file );

    for( int i=0; i<22; i++ )
    {
        // PRINT("i=%d\n",i);
        // get line
        linePointer = fgets( lineTemp, CSV_MAX_LINE_LENGTH, file );

        // linePointer now contains the rowname
        linePointer = strtok( lineTemp, SEP );
        // now the first element
        linePointer = strtok( NULL, SEP );

        if( i==0 )      type = strtod ( linePointer, NULL );
        else if( i==1 ) N    = strtod ( linePointer, NULL );
        else if( i==2 ) P    = strtod ( linePointer, NULL );
        else if( i==3 ) K    = strtod ( linePointer, NULL );
        else if( i==4 )
        {
            nc   = strtod ( linePointer, NULL );
            // printf("N=%d P=%d K=%d nc=%d type=%d\n",N, P, K, nc, type);
            data = new RegressionData( N, P, K, nc, type);
            readCsvAsMatrix( argv[2], data->x, data->N, data->P, data->memory_N );
            readCsvAsMatrix( argv[3], data->y, data->N, data->K, data->memory_N );
            if( data->type >= 5 )
                memcpy( data->yOrg, data->y, data->memory_N * data->K * sizeof(double) );

            if( data->isFusion )
                readCsvAsFusion(argv[4], data);
        }
        else if( i==5 )  data->cSum          = strtod( linePointer, NULL );
        else if( i==6 )  data->alpha         = strtod( linePointer, NULL );
        else if( i==7 )  data->diagonalMoves = (int)strtol( linePointer, NULL, 10 );
        else if( i==8 )  data->useOffset     = (int)strtol( linePointer, NULL, 10 );
        else if( i==9 )  data->useApprox     = (int)strtol( linePointer, NULL, 10 );
        else if( i==10 ) data->precision     = strtod( linePointer, NULL );
        else if( i==11 ) data->algorithm     = (int)strtol( linePointer, NULL, 10 );
        else if( i==12 ) data->verbose       = (int)strtol( linePointer, NULL, 10 );
        else if( i==13 )
        {
            for( int l=0; l<data->N; l++ )
            {
                data->w[l] = strtod( linePointer, NULL );
                if( l != data->N-1 )
                {
                    linePointer = strtok( NULL, SEP );
                    if( linePointer == NULL )
                    {
                        printf("w in settings not of length N\n");
                        exit(1);
                    }
                }
            }
            memcpy( data->wOrg, data->w, data->memory_N * sizeof(double) );
        }
        else if( i==14 )
        {
            for( int l=0; l<data->P; l++ )
            {
                data->u[l] = strtod( linePointer, NULL );
                if( l != data->P-1 )
                {
                    linePointer = strtok( NULL, SEP );
                    if( linePointer == NULL )
                    {
                        printf("u in settings not of length P\n");
                        exit(1);
                    }
                }
            }
        }
        else if( i==15 )
        {
            for( int l=0; l<data->P; l++ )
            {
                data->v[l] = strtod( linePointer, NULL );
                if( l != data->P-1 )
                {
                    linePointer = strtok( NULL, SEP );
                    if( linePointer == NULL )
                    {
                        printf("v in settings not of length P\n");
                        exit(1);
                    }
                }
            }
        }
        else if( i==16 )
        {
            data->lengthLambda = 0;
            do
            {
                sequence[ data->lengthLambda ] = strtod( linePointer, NULL );
                data->lengthLambda++;
                linePointer = strtok( NULL, SEP );
            }
            while( linePointer != NULL && strcmp(linePointer, "") != 0 && strcmp(linePointer, "\n") != 0 );

            #ifdef AVX_VERSION
            data->lambdaSeq = (double*)aligned_alloc( ALIGNMENT,
                                data->lengthLambda * sizeof(double));
            #else
            data->lambdaSeq = (double*)malloc( data->lengthLambda * sizeof(double));
            #endif

            memcpy( data->lambdaSeq, sequence, data->lengthLambda * sizeof(double) );
        }
        else if( i==17 )
        {
            data->lengthGamma = 0;
            do
            {
                sequence[ data->lengthGamma ] = strtod( linePointer, NULL );
                data->lengthGamma++;
                linePointer = strtok( NULL, SEP );
            }
            while( linePointer != NULL && strcmp(linePointer, "") != 0 && strcmp(linePointer, "\n") != 0 );

            #ifdef AVX_VERSION
            data->gammaSeq = (double*)aligned_alloc( ALIGNMENT,
                                data->lengthGamma * sizeof(double));
            #else
            data->gammaSeq = (double*)malloc( data->lengthGamma * sizeof(double));
            #endif

            memcpy( data->gammaSeq, sequence, data->lengthGamma * sizeof(double) );
        }
        else if( i==18 ) data->nFold = (int)strtol( linePointer, NULL, 10 );
        else if( i==19 )
        {
            #ifdef AVX_VERSION
            data->foldid = (int*)aligned_alloc( ALIGNMENT,
                                data->N * sizeof(int));
            #else
            data->foldid = (int*)malloc( data->N * sizeof(int));
            #endif

            for( int l=0; l<data->N; l++ )
            {
                data->foldid[l] = (int)strtol( linePointer, NULL, 10 );
                if( l != data->N-1 )
                {
                    linePointer = strtok( NULL, SEP );
                    if( linePointer == NULL )
                    {
                        printf("foldid in settings not of length N\n");
                        exit(1);
                    }
                }
            }
        }
        else if( i==20 ) data->downScaler = strtod( linePointer, NULL );
        else if( i==21 ) data->cvStop = (int)strtol( linePointer, NULL, 10 );
    }

    fclose(file);
    free(sequence);
    free(lineTemp);

    return data;
}


#endif
