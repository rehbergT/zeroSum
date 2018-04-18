#include "RegressionDataScheme.h"

bool RegressionDataScheme::checkActiveSet(int k) {
    bool isZero = true;
    for (int l = 0; l < K; l++)
        if (beta[INDEX(k, l, memory_P)] != 0.0)
            isZero = false;

    auto it = std::find(activeSet.begin(), activeSet.end(), k);
    bool found = it != activeSet.end();

    if (found && isZero) {
        activeSet.erase(it, activeSet.end());
        return true;
    } else if (!found && !isZero) {
        activeSet.push_back(k);
        return true;
    }
    return false;
}

bool is_zero(double i) {
    return fabs(i) < DBL_EPSILON;
}

void RegressionDataScheme::checkWholeActiveSet() {
    bool isZero;
    int k, l;
    for (int j = activeSet.size() - 1; j >= 0; j--) {
        k = activeSet[j];
        isZero = true;

        for (l = 0; l < K; l++)
            if (beta[INDEX(k, l, memory_P)] != 0.0)
                isZero = false;

        if (isZero)
            activeSet.erase(activeSet.begin() + j);
    }
}

void RegressionDataScheme::doRegression(int seed) {
    if (algorithm == 1) {
        coordinateDescent(seed);
    } else if (algorithm == 2) {
        simulatedAnnealing(seed);
        localSearch(seed);
    } else if (algorithm == 3) {
        localSearch(seed);
    } else if (algorithm == 4) {
        coordinateDescent(seed);
        localSearch(seed);
    }
}

void RegressionDataScheme::calcCoxRegressionD() {
    memset(&status[INDEX(0, 1, memory_N)], 0.0, memory_N * sizeof(int));
    memset(d, 0.0, memory_N * sizeof(double));

    for (int i = 0; i < N - 1; i++) {
        while (yOrg[i] == yOrg[i + 1]) {
            if (status[INDEX(i, 0, memory_N)] == 1 &&
                status[INDEX(i + 1, 0, memory_N)] == 1)
                status[INDEX(i + 1, 1, memory_N)] = 1;
            i++;
        }
    }

    int i = 0;
    while (i < N) {
        if (status[INDEX(i, 0, memory_N)] == 0 || wOrg[i] == 0.0) {
            i++;
            continue;
        }

        d[i] = wOrg[i];
        int k;
        for (k = i + 1; k < N && status[INDEX(k, 1, memory_N)] == 1; k++) {
            if (status[INDEX(k, 0, memory_N)] == 1)
                d[i] += wOrg[k];
        }
        i = k;
    }
}
