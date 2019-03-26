
#include <x86intrin.h>
#include <cstdint>
// N always has to be divideable by the amount of doubles per AVX/AVX512
// register

void interceptMoveKernelAVX2(double* y,
                             double* xb,
                             double* w,
                             uint32_t N,
                             double* result);
void interceptMoveKernelAVX512(double* y,
                               double* xb,
                               double* w,
                               uint32_t N,
                               double* result);

void cvMoveKernelAVX2(double* y,
                      double* xb,
                      double* w,
                      double* xk,
                      double* betak,
                      uint32_t N,
                      double* ak,
                      double* bk);

void cvMoveKernelAVX512(double* y,
                        double* xb,
                        double* w,
                        double* xk,
                        double* betak,
                        uint32_t N,
                        double* ak,
                        double* bk);

void cvMoveZSKernelAVX2(double* y,
                        double* xb,
                        double* w,
                        double* xk,
                        double* xs,
                        double* betak,
                        double* ukus,
                        uint32_t N,
                        double* ak,
                        double* bk);

void cvMoveZSKernelAVX512(double* y,
                          double* xb,
                          double* w,
                          double* xk,
                          double* xs,
                          double* betak,
                          double* ukus,
                          uint32_t N,
                          double* ak,
                          double* bk);

void cvMoveZSKernel2AVX2(double* xb,
                         double* diffk,
                         double* xk,
                         double* diffs,
                         double* xs,
                         uint32_t N);

void cvMoveZSKernel2AVX512(double* xb,
                           double* diffk,
                           double* xk,
                           double* diffs,
                           double* xs,
                           uint32_t N);

void cvMoveZSParallelKernelAVX2(double* xb,
                                double* w,
                                double* xk,
                                double* xs,
                                double* betak,
                                double* ukus,
                                uint32_t N,
                                double* ak,
                                double* bk);

void cvMoveZSParallelKernelAVX512(double* xb,
                                  double* w,
                                  double* xk,
                                  double* xs,
                                  double* betak,
                                  double* ukus,
                                  uint32_t N,
                                  double* ak,
                                  double* bk);

void cdMoveZSRotatedKernelAVX2(double* xm,
                               double* xn,
                               double* xs,
                               double* sinT,
                               double* cosT,
                               double* unum2,
                               uint32_t N,
                               double* res);

void cdMoveZSRotatedKernelAVX512(double* xm,
                                 double* xn,
                                 double* xs,
                                 double* sinT,
                                 double* cosT,
                                 double* unum2,
                                 uint32_t N,
                                 double* res);

void cdMoveZSRotatedKernel2AVX2(double* y,
                                double* xb,
                                double* w,
                                double* tmp,
                                uint32_t N,
                                double* bk);

void cdMoveZSRotatedKernel2AVX512(double* y,
                                  double* xb,
                                  double* w,
                                  double* tmp,
                                  uint32_t N,
                                  double* bk);

void cvMoveZSRotatedKernel3AVX2(double* xb,
                                double* diffn,
                                double* xn,
                                double* diffm,
                                double* xm,
                                double* diffs,
                                double* xs,
                                uint32_t N);

void cvMoveZSRotatedKernel3AVX512(double* xb,
                                  double* diffn,
                                  double* xn,
                                  double* diffm,
                                  double* xm,
                                  double* diffs,
                                  double* xs,
                                  uint32_t N);

void arraySumKernelAVX2(double* a, uint32_t N, double* res);
void arraySumKernelAVX512(double* a, uint32_t N, double* res);

void weightedSquareSumKernelAVX2(double* a, double* b, uint32_t N, double* res);
void weightedSquareSumKernelAVX512(double* a,
                                   double* b,
                                   uint32_t N,
                                   double* res);

void weightedResidualSquareSumKernelAVX2(double* a,
                                         double* b,
                                         double* c,
                                         uint32_t N,
                                         double* res);
void weightedResidualSquareSumKernelAVX512(double* a,
                                           double* b,
                                           double* c,
                                           uint32_t N,
                                           double* res);

void squareWeightedSumKernelAVX2(double* a, double* b, uint32_t N, double* res);
void squareWeightedSumKernelAVX512(double* a,
                                   double* b,
                                   uint32_t N,
                                   double* res);

void a_sub_bKernelAVX2(double* a, double* b, double* c, uint32_t N);
void a_sub_bKernelAVX512(double* a, double* b, double* c, uint32_t N);

void a_add_scalar_bKernelAVX2(double* a, double* b, uint32_t N);
void a_add_scalar_bKernelAVX512(double* a, double* b, uint32_t N);