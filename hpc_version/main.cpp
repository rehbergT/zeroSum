#include <mpi.h>
#include <omp.h>
#include <cstdio>
#include <ctime>

#include "../zeroSum/src/zeroSum.h"
#include "csv_read_write.h"

#ifdef G_PROF
#include <gperftools/profiler.h>
#endif

int main(int32_t argc, char** argv) {
    int32_t mpi_processes, mpi_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

#ifdef DEBUG
    printf("MPI rank %d of %d MPI process\n", mpi_rank, mpi_processes);
#endif

    struct timespec ts0, ts1;
    zeroSum* data = nullptr;

    if (mpi_rank == MASTER) {
        clock_gettime(CLOCK_REALTIME, &ts0);
    }

    data = readData(argc, argv);
    readSaves(argv[argc - 2], argv[argc - 1], *data);

#ifdef DEBUG
    printf("mpi_rank:%d N:%d P:%d K:%d type:%d nc:%d nfold:%d memory_N:%d\n",
           mpi_rank, data->N, data->P, data->K, data->type, data->nc,
           data->nFold, data->memory_N);
    printf(
        "mpi_rank:%d cSum=%e alpha=%e diag:%d off:%d app:%d, precision:%e \n"
        "verbose %d downscaler %e cvstop: %d threads: %d\n",
        mpi_rank, data->cSum, data->alpha, data->rotatedUpdates,
        data->useIntercept, data->useApprox, data->precision, data->verbose,
        data->downScaler, data->cvStop, data->threads);
    if (data->verbose)
        printf("mpi_rank:%d gammalength: %lu lambdaLength: %lu\n", mpi_rank,
               data->gammaSeq.size(), data->lambdaSeq.size());
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    int32_t rest = data->gammaSeq.size() % mpi_processes;
    int32_t toDoEveryone = (data->gammaSeq.size() - rest) / mpi_processes;
    int32_t toDoProcess = toDoEveryone;
    if (mpi_rank < rest)
        toDoProcess++;

    uint32_t start = 0, end = 0;
    for (int32_t i = 0; i < mpi_processes; i++) {
        end = start + toDoEveryone;
        if (i < rest)
            end++;
        if (i == mpi_rank)
            break;
        start = end;
    }

    std::vector<double> gammaSeqOrg = data->gammaSeq;
    data->gammaSeq.clear();

#ifdef DEBUG
    printf("start: %d ende: %d\n", start, end);
#endif
    for (uint32_t i = start; i < end; i++)
        data->gammaSeq.push_back(gammaSeqOrg[i]);
#ifdef DEBUG
    data->printMatrix(data->gammaSeq.data(), data->gammaSeq.size(), 1);
#endif

#ifdef G_PROF
    printf("Startings GPROG\n");
    ProfilerStart("profile.log");
#endif
    data->doCVRegression(argv[argc - 2], argv[argc - 1], mpi_rank);
#ifdef G_PROF
    ProfilerStop();
#endif

    if (mpi_rank == 0) {
        clock_gettime(CLOCK_REALTIME, &ts1);
        double timet =
            (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
        if (data->verbose)
            printf("DONE\t runtime: %f s\n", timet);
    }

    delete (data);

    MPI_Finalize();
    return 0;
}
