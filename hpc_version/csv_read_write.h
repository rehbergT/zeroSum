#ifndef CSV_READ_WRITE_H
#define CSV_READ_WRITE_H

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mpi.h>
#include "../zeroSum/src/RegressionCV.h"

#define CSV_MAX_LINE_LENGTH 1e8
#define MAX_SEQUENCE 1e5
#define SEP ","
#define MASTER 0

void    readCsvAsMatrix( char* path, double* matrix, int N, int M, int mN );
void    readCsvAsFusion( char* path, RegressionData& data );
double* readCsvSave(     char* path, int* N, int* M );
void    readSaves(       char* path, char* name, RegressionData& data);

RegressionData* readRegressionData( int argc, char **argv );
RegressionData* MPI_Bcast_RegressionData( RegressionData *data, int mpi_rank );

#endif
