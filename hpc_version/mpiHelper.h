#ifndef MPIHELPER
#define MPIHELPER

RegressionData* MPI_Bcast_RegressionData( RegressionData* data, int mpi_rank )
{
    int N=0, P=0, K=0, nc=0, type=0;

    if( mpi_rank == MASTER )
    {
        N    = data->N;
        P    = data->P;
        K    = data->K;
        nc   = data->nc;
        type = data->type;
    }

    MPI_Bcast( &N,    1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast( &P,    1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast( &K,    1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast( &nc,   1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast( &type, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    if( mpi_rank != MASTER)
        data = new RegressionData( N, P, K, nc, type);

    MPI_Bcast( &data->nFold,         1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast( &data->diagonalMoves, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast( &data->useOffset,     1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast( &data->useApprox,     1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast( &data->algorithm,     1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast( &data->verbose,       1, MPI_INT, MASTER, MPI_COMM_WORLD);

    // int lengthGamma;
    // if( mpi_rank == MASTER) lengthGamma = data->lengthGamma;
    // MPI_Bcast( &lengthGamma, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    // if( mpi_rank != MASTER) data->lengthGamma = 7;

    MPI_Bcast( &data->lengthGamma,  1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast( &data->lengthLambda, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast( &data->cvStop,       1, MPI_INT, MASTER, MPI_COMM_WORLD);

    MPI_Bcast( &data->cSum,       1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Bcast( &data->alpha,      1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Bcast( &data->precision,  1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Bcast( &data->downScaler, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    MPI_Bcast( data->x, data->memory_N * data->P, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Bcast( data->y, data->memory_N * data->K, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Bcast( data->w, data->memory_N,           MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Bcast( data->v, data->memory_P,           MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Bcast( data->u, data->memory_P,           MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    if( mpi_rank != MASTER)
    {
        #ifdef AVX_VERSION
        data->lambdaSeq = (double*)aligned_alloc( ALIGNMENT,
                            data->lengthLambda * sizeof(double));
        data->gammaSeq  = (double*)aligned_alloc( ALIGNMENT,
                            data->lengthGamma * sizeof(double));
        data->foldid    = (int*)aligned_alloc( ALIGNMENT,
                            data->N * sizeof(int));
        #else
        data->lambdaSeq = (double*)malloc( data->lengthLambda * sizeof(double));
        data->gammaSeq  = (double*)malloc( data->lengthGamma * sizeof(double));
        data->foldid    = (int*)malloc( data->N * sizeof(int));
        #endif

    }

    MPI_Bcast( data->lambdaSeq, data->lengthLambda, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Bcast( data->gammaSeq,  data->lengthGamma,  MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Bcast( data->foldid,    data->N,            MPI_INT,    MASTER, MPI_COMM_WORLD);

    if( data->isFusion)
    {
        int i, numElements;
        double x;

        for( int j=0; j<data->P; j++)
        {
            numElements = 0;
            if( mpi_rank == MASTER)
            {
                struct fusionKernel * currentEl = data->fusionKernel[j];
                while( currentEl != NULL )
                {
                    numElements++;
                    currentEl = currentEl->next;
                }
            }

            MPI_Bcast( &numElements, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
            // printf("Numelements %d\n", numElements );

            struct fusionKernel * currentEl;
            if( mpi_rank == MASTER)
                currentEl = data->fusionKernel[j];

            for( int ii=0; ii<numElements; ii++)
            {
                if( mpi_rank == MASTER)
                {
                    i = currentEl->i;
                    x = currentEl->value;
                    currentEl = currentEl->next;
                }

                MPI_Bcast( &i, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
                MPI_Bcast( &j, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
                MPI_Bcast( &x, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

                if( mpi_rank != MASTER)
                    data->fusionKernel[j] = appendElement( data->fusionKernel[j], i, x);
            }
        }
    }

    if( mpi_rank != MASTER)
    {
        memcpy( data->wOrg, data->w, data->memory_N * sizeof(double) );
        if( data->type >= 5 )
            memcpy( data->yOrg, data->y, data->memory_N * data->K * sizeof(double) );
    }

    // printf("mpi_rank:%d N:%d P:%d K:%d type:%d nc:%d nfold:%d memory_N:%d\n", mpi_rank, data->N,
    //     data->P, data->K, data->type, data->nc, data->nFold, data->memory_N );
    // printf("mpi_rank:%d cSum=%e alpha=%e diag:%d off:%d app:%d, precision:%e\n",
    //     mpi_rank, data->cSum, data->alpha, data->diagonalMoves, data->useOffset,
    //     data->useApprox, data->precision);
    //
    // printf("mpi_rank:%d gammalength: %d lambdaLength: %d\n", mpi_rank, data->lengthGamma,
    //     data->lengthLambda);

    // if( mpi_rank == MASTER )
    // {
    //     printMatrixColWise( data->x, data->N, data->P);
    //     printMatrixColWise( data->y, data->N, data->K);
    //     printMatrixColWise( data->w, data->N, 1);
    //     printMatrixColWise( data->u, data->P, 1);
    //     printMatrixColWise( data->v, data->P, 1);
    //     printMatrixColWise( data->lambdaSeq, data->lengthLambda, 1);
    //     printMatrixColWise( data->gammaSeq, data->lengthGamma, 1);
    //     if( data->isFusion) printSparseFusion( data->fusionKernel, data->nc, data->P);
    // }
    // MPI_Barrier(MPI_COMM_WORLD);
    //
    // if( mpi_rank == MASTER+1 )
    // {
    //     printMatrixColWise( data->x, data->N, data->P);
    //     printMatrixColWise( data->y, data->N, data->K);
    //     printMatrixColWise( data->w, data->N, 1);
    //     printMatrixColWise( data->u, data->P, 1);
    //     printMatrixColWise( data->v, data->P, 1);
    //     printMatrixColWise( data->lambdaSeq, data->lengthLambda, 1);
    //     printMatrixColWise( data->gammaSeq, data->lengthGamma, 1);
    //     if( data->isFusion) printSparseFusion( data->fusionKernel, data->nc, data->P);
    // }


    return data;
}




#endif
