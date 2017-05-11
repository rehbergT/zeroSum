#include <omp.h>
#include <mpi.h>
#include <ctime>
#include <cstdio>

#include "../zeroSum/src/RegressionCV.h"
#include "csv_read_write.h"


void printMatrixColWise( double* matrix, int N, int P )
{
    for( int n=0; n<N; ++n )
    {
        for( int p=0; p<P; ++p )
           PRINT("%d  %+.3e\t", INDEX(n,p,N) , matrix[INDEX(n,p,N)]);
        PRINT("\n");
    }
}

int main( int argc, char **argv )
{
    int mpi_processes, mpi_rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &mpi_processes );
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );

    printf( "MPI rank %d of %d MPI process\n",
       mpi_rank, mpi_processes );

    struct timespec ts0, ts1;
    RegressionData* data = nullptr;

    if( mpi_rank == MASTER )
    {
        data = readRegressionData( argc, argv );
        readSaves( argv[argc-2], argv[argc-1], *data );
        clock_gettime(CLOCK_REALTIME , &ts0);
    }

    data = MPI_Bcast_RegressionData( data, mpi_rank);

    printf("mpi_rank:%d N:%d P:%d K:%d type:%d nc:%d nfold:%d memory_N:%d\n", mpi_rank, data->N,
         data->P, data->K, data->type, data->nc, data->nFold, data->memory_N );
    printf("mpi_rank:%d cSum=%e alpha=%e diag:%d off:%d app:%d, precision:%e verbose %d downscaler %e cvstop: %d\n",
     mpi_rank, data->cSum, data->alpha, data->diagonalMoves, data->useOffset,
     data->useApprox, data->precision, data->verbose, data->downScaler, data->cvStop);

    printf("mpi_rank:%d gammalength: %d lambdaLength: %d\n", mpi_rank, data->lengthGamma,
     data->lengthLambda);
    MPI_Barrier(MPI_COMM_WORLD);

  //     if( mpi_rank == MASTER )
  //     {
  //         printMatrixColWise( data->x, data->N, data->P);
  //         printMatrixColWise( data->y, data->N, data->K);
  //         printMatrixColWise( data->w, data->N, 1);
  //         printMatrixColWise( data->u, data->P, 1);
  //         printMatrixColWise( data->v, data->P, 1);
  //         printMatrixColWise( data->lambdaSeq, data->lengthLambda, 1);
  //         printMatrixColWise( data->gammaSeq, data->lengthGamma, 1);
  // //         if( data->isFusion) printSparseFusion( data->fusionKernel, data->nc, data->P);
  //     }
  //     MPI_Barrier(MPI_COMM_WORLD);
  //
  //     if( mpi_rank == MASTER+1 )
  //     {
  // //         printMatrixColWise( data->x, data->N, data->P);
  // //         printMatrixColWise( data->y, data->N, data->K);
  // //         printMatrixColWise( data->w, data->N, 1);
  // //         printMatrixColWise( data->u, data->P, 1);
  // //         printMatrixColWise( data->v, data->P, 1);
  //         printMatrixColWise( data->lambdaSeq, data->lengthLambda, 1);
  //         printMatrixColWise( data->gammaSeq, data->lengthGamma, 1);
  // //         if( data->isFusion) printSparseFusion( data->fusionKernel, data->nc, data->P);
  //     }

    int rest = data->lengthGamma % mpi_processes;
    int toDoEveryone = ( data->lengthGamma - rest ) / mpi_processes;
    int toDoProcess = toDoEveryone;
    if( mpi_rank < rest )
        toDoProcess++;

    int start = 0, end = 0;
    for( int i=0; i<mpi_processes; i++ )
    {
        end = start + toDoEveryone;
        if( i < rest ) end++;
        if( i == mpi_rank ) break;
        start = end;
    }

    double seed = 0.0;

    double* gammaSeqOrg = data->gammaSeq;
    data->gammaSeq = &data->gammaSeq[start];
    data->lengthGamma = end-start;

    RegressionCV cvRegression( *data );
    cvRegression.doCVRegression( seed, argv[argc-2], argv[argc-1], mpi_rank );
    if( mpi_rank == 0 )
    {
        clock_gettime(CLOCK_REALTIME , &ts1);
        double timet = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
        printf("DONE\t runtime: %f min\n", timet/60.0);
    }

    free(data->lambdaSeq);
    free(gammaSeqOrg);
    free(data->foldid);
    delete(data);

    MPI_Finalize();
    return 0;
}
