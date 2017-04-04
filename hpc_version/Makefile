CC    = g++
MPICC = mpicxx
LIBS = -lm

ARCH = native

CXXFLAGS = -std=c++11 -fopenmp -O3 -flto -march=$(ARCH) -mtune=$(ARCH)
WARNINGS = -Wall -Wextra -Wunreachable-code -Werror=format-security

ZEROSUM_SRC = ../zeroSum/src

OBJ  = CvRegression.o \
	   CvRegressionData.o \
	   doRegression.o \
	   fusionKernel.o \
	   mathHelpers.o \
	   RegressionData.o \
	   RegressionDataSchemeActiveSet.o \
	   RegressionDataSchemeApproximation.o \
	   RegressionDataSchemeCdMoves.o \
	   RegressionDataSchemeConstructors.o \
	   RegressionDataSchemeCoordinateDescent.o \
	   RegressionDataSchemeCostFunction.o \
	   RegressionDataSchemeDebugChecks.o \
	   RegressionDataSchemeLocalSearch.o \
	   RegressionDataSchemeLsSaMoves.o \
	   RegressionDataSchemeMultinomialParameterAmbiguity.o \
	   RegressionDataSchemePredict.o \
	   RegressionDataSchemeSimulatedAnnealing.o


_DEPS = CvRegression.h \
		CvRegressionData.h \
		fusionKernel.h \
		mathHelpers.h \
		RegressionData.h \
		RegressionDataScheme.h \
		regressions.h

DEPS = $(patsubst %,$(ZEROSUM_SRC)/%,$(_DEPS))

%.o: $(ZEROSUM_SRC)/%.cpp $(DEPS)
	$(CC) $(CXXFLAGS) $(WARNINGS) -c -o $@ $<

all: $(OBJ)
	$(MPICC) $(CXXFLAGS) $(WARNINGS) -c main.cpp
	$(MPICC) main.o $^ $(LIBS) -fopenmp -o zeroSum

.PHONY: clean
clean:
	rm -f *.o
	rm -f zeroSum
	rm -f *.gch

