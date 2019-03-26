
#include "csv_read_write.h"

void readCsvAsMatrix(char* path,
                     double* matrix,
                     uint32_t N,
                     uint32_t M,
                     uint32_t mN) {
    uint32_t n, m;  // Matrix dimensions
    uint32_t i, j;  // loop variables
    FILE* file = fopen(path, "r");

    if (file == NULL) {
        printf("Error: file pointer is null\n");
        exit(1);
    }

    // define a temp char array for storing a line of the csv file
    char* lineTemp = (char*)malloc(CSV_MAX_LINE_LENGTH * sizeof(char));
    char* linePointer = NULL;

    // get dimensions of the matrix
    n = 0;
    m = 0;
    while (fgets(lineTemp, CSV_MAX_LINE_LENGTH, file)) {
        if (n == 1) {
            linePointer = strtok(lineTemp, SEP);

            while (linePointer != NULL) {
                linePointer = strtok(NULL, SEP);
                ++m;
            }
        }
        ++n;
    }
    // reduce by one because of header line
    --n;
    // reduce by one because of row names
    --m;
    // reset file pointer
    rewind(file);
    if (n != N || m != M) {
        printf("Error: wrong dimensions of csv file\n");
        exit(1);
    }

    i = 0;
    j = 0;

    // first line are colnames -> read & drop it
    linePointer = fgets(lineTemp, CSV_MAX_LINE_LENGTH, file);

    while (fgets(lineTemp, CSV_MAX_LINE_LENGTH, file)) {
        linePointer = strtok(lineTemp, SEP);
        for (j = 0; j < m; ++j) {
            linePointer = strtok(NULL, SEP);
            matrix[INDEX(i, j, mN)] = strtod(linePointer, NULL);
        }
        i++;
    }

    fclose(file);
    free(lineTemp);
}

void readCsvAsIntMatrix(char* path,
                        uint32_t* matrix,
                        uint32_t N,
                        uint32_t M,
                        uint32_t mN) {
    uint32_t n, m;  // Matrix dimensions
    uint32_t i, j;  // loop variables
    FILE* file = fopen(path, "r");

    if (file == NULL) {
        printf("Error: file pointer is null\n");
        exit(1);
    }

    // define a temp char array for storing a line of the csv file
    char* lineTemp = (char*)malloc(CSV_MAX_LINE_LENGTH * sizeof(char));
    char* linePointer = NULL;

    // get dimensions of the matrix
    n = 0;
    m = 0;
    while (fgets(lineTemp, CSV_MAX_LINE_LENGTH, file)) {
        ++n;
        if (n == 1) {
            linePointer = strtok(lineTemp, SEP);

            while (linePointer != NULL) {
                linePointer = strtok(NULL, SEP);
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
    if (n != N || m != M) {
        printf("Error: wrong dimensions of csv file\n");
        exit(1);
    }

    i = 0;
    j = 0;

    // first line are colnames -> read & drop it
    linePointer = fgets(lineTemp, CSV_MAX_LINE_LENGTH, file);

    while (fgets(lineTemp, CSV_MAX_LINE_LENGTH, file)) {
        linePointer = strtok(lineTemp, SEP);
        for (j = 0; j < m; ++j) {
            linePointer = strtok(NULL, SEP);
            matrix[INDEX(i, j, mN)] = strtod(linePointer, NULL);
        }
        i++;
    }

    fclose(file);
    free(lineTemp);
}

void readCsvAsFusion(char* path, zeroSum& data) {
    FILE* file = fopen(path, "r");

    if (file == NULL) {
        printf("Error: file pointer is null\n");
        exit(1);
    }

    // define a temp char array for storing a line of the csv file
    char* lineTemp = (char*)malloc(CSV_MAX_LINE_LENGTH * sizeof(char));
    char* linePointer = fgets(lineTemp, CSV_MAX_LINE_LENGTH, file);

    uint32_t i, j;
    double x;

    while (fgets(lineTemp, CSV_MAX_LINE_LENGTH, file)) {
        // two times since first element is the rowname
        linePointer = strtok(lineTemp, SEP);
        linePointer = strtok(NULL, SEP);
        i = (uint32_t)strtol(linePointer, NULL, 10);
        linePointer = strtok(NULL, SEP);
        j = (uint32_t)strtol(linePointer, NULL, 10);
        linePointer = strtok(NULL, SEP);
        x = strtod(linePointer, NULL);
        // printf("i=%d j=%d x=%e\n",i,j,x);
        data.fusionKernel[j] = appendElement(data.fusionKernel[j], i, x);
    }

    fclose(file);
    free(lineTemp);
}

double* readCsvSave(char* path, uint32_t* N, uint32_t* M) {
    uint32_t n, m;  // Matrix dimensions
    uint32_t i, j;  // loop variables
    FILE* file = fopen(path, "r");

    // define a temp char array for storing a line of the csv file
    char* lineTemp = (char*)malloc(CSV_MAX_LINE_LENGTH * sizeof(char));
    char* linePointer = NULL;

    // get dimensions of the matrix
    n = 0;
    m = 0;
    while (fgets(lineTemp, CSV_MAX_LINE_LENGTH, file)) {
        ++n;
        if (n == 1) {
            linePointer = strtok(lineTemp, SEP);

            while (linePointer != NULL) {
                linePointer = strtok(NULL, SEP);
                ++m;
            }
        }
    }

    rewind(file);
    *N = n;
    *M = m;
    i = 0;
    j = 0;

    double* matrix = (double*)malloc(n * m * sizeof(double));

    while (fgets(lineTemp, CSV_MAX_LINE_LENGTH, file)) {
        for (j = 0; j < m; ++j) {
            if (j == 0)
                linePointer = strtok(lineTemp, SEP);
            else
                linePointer = strtok(NULL, SEP);
            matrix[INDEX(i, j, n)] = strtod(linePointer, NULL);
        }
        i++;
    }

    fclose(file);
    free(lineTemp);
    return (matrix);
}

uint32_t getMinIndex(double* a, uint32_t N) {
    uint32_t min = 0;
    for (uint32_t i = 1; i < N; ++i)
        if (a[min] > a[i])
            min = i;

    return min;
}

void readSaves(char* path, char* name, zeroSum& data) {
    FILE* file;
    char filePath[MAX_LINE_LENGTH];
    uint32_t fileNumber = 0;

    // find the smallest commen lambda in each
    double lambdaCommon = -2.0;

    while (true) {
        snprintf(filePath, sizeof(filePath), "%s%s_%d_stats.csv", path, name,
                 fileNumber);

        file = fopen(filePath, "r");
        if (file == NULL)
            break;
        fclose(file);

        printf("found: %s\n", filePath);

        uint32_t rows, cols;
        double* content = readCsvSave(filePath, &rows, &cols);

        uint32_t min = getMinIndex(&content[INDEX_COL(1, rows)], rows);

        if (fileNumber == 0)
            lambdaCommon = content[INDEX(min, 1, rows)];
        else if (content[INDEX(min, 1, rows)] > lambdaCommon)
            lambdaCommon = content[INDEX(min, 1, rows)];

        fileNumber++;
        free(content);
    }

    if (lambdaCommon != -2.0) {
        printf("last calculated lambda: %f\n", lambdaCommon);
        size_t index = -1;
        for (size_t i = 0; i < data.lambdaSeq.size(); i++) {
            if (fabs(data.lambdaSeq[i] - lambdaCommon) < DBL_EPSILON * 100)
                index = i;
        }

        if (index == data.lambdaSeq.size() - 1) {
            printf("Stopped! Everythings seems to be calculated\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return;
        }

        std::vector<double> lambdaSeqOrg = data.lambdaSeq;
        data.lambdaSeq.clear();

        for (size_t i = index + 1; i < lambdaSeqOrg.size(); i++)
            data.lambdaSeq.push_back(lambdaSeqOrg[i]);

        printf("Found save files starting from lambda: %f\n", lambdaCommon);
    }
}

zeroSum* readData(uint32_t argc, char** argv) {
    if (argc != 6 && argc != 7 && argc != 8) {
        printf("Error: only %d arguments\n", argc);
        exit(1);
    }
    zeroSum* data;

    char* lineTemp = (char*)malloc(CSV_MAX_LINE_LENGTH * sizeof(char));
    double* sequence = (double*)malloc(MAX_SEQUENCE * sizeof(double));

    char* linePointer = NULL;

    FILE* file;
    file = fopen(argv[1], "r");

    if (file == NULL) {
        printf("Error: file pointer is null\n");
        exit(1);
    }

    uint32_t N = 0, P = 0, K = 0, nc = 0, type = 0, useZeroSum = 0,
             useFusion = 0, useIntercept = 0, useApprox = 0, useCentering = 0,
             useStandardization = 0, usePolish = 0, rotatedUpdates = 0,
             algorithm = 0, nFold = 0, cvStop = 0, verbose = 0, threads = 0,
             seed = 0;
    double precision = 0.0, cSum = 0.0, alpha = 0.0, downScaler = 0.0;

    // first line are colnames -> read & drop it
    linePointer = fgets(lineTemp, CSV_MAX_LINE_LENGTH, file);

    for (uint32_t line = 0; line < 29; line++) {
        // printf("line=%d\n",line);
        // get line
        linePointer = fgets(lineTemp, CSV_MAX_LINE_LENGTH, file);

        // linePointer now contains the rowname
        linePointer = strtok(lineTemp, SEP);
        // now the first element
        linePointer = strtok(NULL, SEP);

        if (line == 0)
            N = strtod(linePointer, NULL);
        else if (line == 1)
            P = strtod(linePointer, NULL);
        else if (line == 2)
            K = strtod(linePointer, NULL);
        else if (line == 3)
            nc = strtod(linePointer, NULL);
        else if (line == 4)
            type = strtod(linePointer, NULL);
        else if (line == 5)
            useZeroSum = strtod(linePointer, NULL);
        else if (line == 6)
            useFusion = strtod(linePointer, NULL);
        else if (line == 7)
            useIntercept = strtod(linePointer, NULL);
        else if (line == 8)
            useApprox = strtod(linePointer, NULL);
        else if (line == 9)
            useCentering = strtod(linePointer, NULL);
        else if (line == 10)
            useStandardization = strtod(linePointer, NULL);
        else if (line == 11)
            usePolish = strtod(linePointer, NULL);
        else if (line == 12)
            rotatedUpdates = strtod(linePointer, NULL);
        else if (line == 13)
            precision = strtod(linePointer, NULL);
        else if (line == 14)
            algorithm = strtod(linePointer, NULL);
        else if (line == 15)
            nFold = strtod(linePointer, NULL);
        else if (line == 16)
            cvStop = strtod(linePointer, NULL);
        else if (line == 17)
            verbose = strtod(linePointer, NULL);
        else if (line == 18)
            cSum = strtod(linePointer, NULL);
        else if (line == 19)
            alpha = strtod(linePointer, NULL);
        else if (line == 20)
            downScaler = strtod(linePointer, NULL);
        else if (line == 21)
            threads = strtod(linePointer, NULL);
        else if (line == 22) {
            seed = strtod(linePointer, NULL);
            if (verbose)
                printf(
                    "N=%d P=%d K=%d nc=%d type=%d useZeroSum=%d useFusion=%d "
                    "nFold=%d\n",
                    N, P, K, nc, type, useZeroSum, useFusion, nFold);
            data = new zeroSum(N, P, K, nc, type, useZeroSum, useFusion,
                               useIntercept, useApprox, useCentering,
                               useStandardization, usePolish, rotatedUpdates,
                               precision, algorithm, nFold, cvStop, verbose,
                               cSum, alpha, downScaler, threads, seed);
            readCsvAsMatrix(argv[2], data->x, data->N, data->P, data->memory_N);
            readCsvAsMatrix(argv[3], data->y, data->N, data->K, data->memory_N);

            if (type == zeroSum::types::cox)
                readCsvAsIntMatrix(argv[4], data->status, data->N, data->K,
                                   data->memory_N);

            for (uint32_t f = 1; f < data->nFold1; f++)
                memcpy(
                    &data->y[INDEX_TENSOR_COL(0, f, data->memory_N, data->K)],
                    data->y, data->memory_N * data->K * sizeof(double));

            memcpy(data->yOrg, data->y,
                   data->memory_N * data->K * sizeof(double));

            if (data->useFusion) {
                if (type == zeroSum::types::cox)
                    readCsvAsFusion(argv[5], *data);
                else
                    readCsvAsFusion(argv[4], *data);
            }
        } else if (line == 23) {
            for (uint32_t i = 0; i < data->N; i++) {
                data->w[i] = strtod(linePointer, NULL);
                if (i != data->N - 1) {
                    linePointer = strtok(NULL, SEP);
                    if (linePointer == NULL) {
                        printf("w in settings not of length N\n");
                        exit(1);
                    }
                }
            }

            for (uint32_t f = 0; f < data->nFold1; f++) {
                for (uint32_t l = 0; l < data->K; ++l) {
                    if (f == 0 && l == 0)
                        continue;
                    uint32_t iiN =
                        INDEX_TENSOR_COL(l, f, data->memory_N, data->K);
                    memcpy(&data->w[iiN], data->w, data->N * sizeof(double));
                }

                uint32_t iiF = INDEX_COL(f, data->memory_N);
                memcpy(&data->wOrg[iiF], data->w, data->N * sizeof(double));
                memcpy(&data->wCV[iiF], data->w, data->N * sizeof(double));
            }
        } else if (line == 24) {
            for (uint32_t i = 0; i < data->P; i++) {
                data->u[i] = strtod(linePointer, NULL);
                if (i != data->P - 1) {
                    linePointer = strtok(NULL, SEP);
                    if (linePointer == NULL) {
                        printf("u in settings not of length P\n");
                        exit(1);
                    }
                }
            }
        } else if (line == 25) {
            for (uint32_t i = 0; i < data->P; i++) {
                data->v[i] = strtod(linePointer, NULL);
                if (i != data->P - 1) {
                    linePointer = strtok(NULL, SEP);
                    if (linePointer == NULL) {
                        printf("v in settings not of length P\n");
                        exit(1);
                    }
                }
            }
            for (uint32_t f = 1; f < data->nFold1; f++)
                memcpy(&data->v[INDEX_COL(f, data->memory_P)], data->v,
                       data->P * sizeof(double));
        } else if (line == 26) {
            do {
                data->lambdaSeq.push_back(strtod(linePointer, NULL));
                for (uint32_t f = 0; f < nFold + 1; f++)
                    data->lambda[f] = data->lambdaSeq[0];
                linePointer = strtok(NULL, SEP);
            } while (linePointer != NULL && strcmp(linePointer, "") != 0 &&
                     strcmp(linePointer, "\n") != 0);
        } else if (line == 27) {
            do {
                data->gammaSeq.push_back(strtod(linePointer, NULL));
                linePointer = strtok(NULL, SEP);
            } while (linePointer != NULL && strcmp(linePointer, "") != 0 &&
                     strcmp(linePointer, "\n") != 0);
        } else if (line == 28) {
            for (uint32_t i = 0; i < data->N; i++) {
                data->foldid.push_back((uint32_t)strtol(linePointer, NULL, 10));
                if (i != data->N - 1) {
                    linePointer = strtok(NULL, SEP);
                    if (linePointer == NULL) {
                        printf("foldid in settings not of length N\n");
                        exit(1);
                    }
                }
            }
        }
    }

    fclose(file);
    free(sequence);
    free(lineTemp);

    return data;
}
