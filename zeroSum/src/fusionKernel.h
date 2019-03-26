#ifndef FUSIONKERNEL_H
#define FUSIONKERNEL_H

#include <stdlib.h>
#include <cstdint>

struct fusionKernel {
    uint32_t i;
    double value;
    struct fusionKernel* next;
};

struct fusionKernel* appendElement(struct fusionKernel* preElement,
                                   uint32_t i,
                                   double value);

#endif /* REGRESSIONDATA_H */
