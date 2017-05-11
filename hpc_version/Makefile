CC    = g++
MPICC = mpicxx
LIBS = -lm

ARCH = native

CXXFLAGS = -std=c++11 -fopenmp -O3 -flto -march=$(ARCH) -mtune=$(ARCH)
WARNINGS = -Wall -Wextra -Wunreachable-code -Werror=format-security

ZEROSUM_SRC = ../zeroSum/src

OBJ  = main.o \
	   csv_read_write.o \
	   CvRegressionData.o \
	   RegressionCV.o \
	   fusionKernel.o \
	   mathHelpers.o \
	   RegressionData.o \
	   RegressionDataScheme.o \
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


_DEPS = RegressionCV.h \
		CvRegressionData.h \
		fusionKernel.h \
		mathHelpers.h \
		RegressionData.h \
		RegressionDataScheme.h \
		RegressionCV.h

DEPS = $(patsubst %,$(ZEROSUM_SRC)/%,$(_DEPS))

all: $(OBJ)
	$(MPICC) -flto $^ $(LIBS) -fopenmp -o zeroSum

main.o: main.cpp
	$(MPICC) $(CXXFLAGS) $(WARNINGS) -c main.cpp

csv_read_write.o: csv_read_write.cpp csv_read_write.h
	$(MPICC) $(CXXFLAGS) $(WARNINGS) -c csv_read_write.cpp

%.o: $(ZEROSUM_SRC)/%.cpp $(DEPS)
	$(CC) $(CXXFLAGS) $(WARNINGS) -c -o $@ $<


.PHONY: clean
clean:
	rm -f *.o
	rm -f zeroSum
	rm -f *.gch
