#include "zeroSum.h"

void zeroSum::printMatrix(double* x, uint32_t N, uint32_t M) {
    for (uint32_t i = 0; i < N; i++) {
        PRINT("%02d ", i);
        for (uint32_t j = 0; j < M; j++) {
            PRINT("%+.6e ", x[INDEX(i, j, N)]);
        }
        PRINT("\n");
    }
}

void zeroSum::printIntMatrix(uint32_t* x, uint32_t N, uint32_t M) {
    for (uint32_t i = 0; i < N; i++) {
        PRINT("%02d ", i);
        for (uint32_t j = 0; j < M; j++) {
            PRINT("%d ", x[INDEX(i, j, N)]);
        }
        PRINT("\n");
    }
}

void zeroSum::printSparseMatrix(struct fusionKernel** x, uint32_t M) {
    struct fusionKernel* currEl;
    for (uint32_t j = 0; j < M; j++) {
        currEl = x[j];
        while (currEl != NULL) {
            PRINT("i=%d j=%d value=%f\n", currEl->i, j, currEl->value);
            currEl = currEl->next;
        }
    }
}

void zeroSum::activeSetPrint(uint32_t fold) {
    PRINT("Active set (fold %d) totalsize: %lu\n", fold,
          activeSet[fold].size());
    for (uint32_t i = 0; i < P; i++) {
        if (activeSetContains(fold, i)) {
            PRINT("beta[%d]=%e, %d\n", i,
                  beta[INDEX_TENSOR(i, 0, fold, memory_P, K)], i);
        } else {
            PRINT("beta[%d]=%e, --\n", i,
                  beta[INDEX_TENSOR(i, 0, fold, memory_P, K)]);
        }
    }
}