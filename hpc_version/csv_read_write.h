#ifndef CSV_READ_WRITE_H
#define CSV_READ_WRITE_H

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "../zeroSum/src/zeroSum.h"

#define CSV_MAX_LINE_LENGTH 1e8
#define MAX_SEQUENCE 1e5
#define SEP ","
#define MASTER 0

void readCsvAsMatrix(char* path,
                     double* matrix,
                     uint32_t N,
                     uint32_t M,
                     uint32_t mN);
void readCsvAsFusion(char* path, zeroSum& data);
double* readCsvSave(char* path, uint32_t* N, uint32_t* M);
void readSaves(char* path, char* name, zeroSum& data);

zeroSum* readData(uint32_t argc, char** argv);
zeroSum* MPI_Bcast_Data(zeroSum* data, uint32_t mpi_rank);

#endif
