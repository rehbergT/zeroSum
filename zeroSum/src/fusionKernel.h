#ifndef FUSIONKERNEL_H
#define FUSIONKERNEL_H

#include <stdlib.h>

struct fusionKernel {
    int i;
    double value;
    struct fusionKernel* next;
};

struct fusionKernel* appendElement(struct fusionKernel* preElement,
                                   int i,
                                   double value);

#endif /* REGRESSIONDATA_H */
