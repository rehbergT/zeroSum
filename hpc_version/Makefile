CC    = g++
MPICC = mpicxx
LIBS = -lm -lblas -lpthread
ARCH = native
CXXFLAGS = -std=c++11 -O3 -flto -march=$(ARCH) -mtune=$(ARCH) -DBUILD_WITH_AVX512 -DBUILD_WITH_AVX2
WARNINGS = -Wall -Wextra -Wunreachable-code -Werror=format-security -pedantic-errors

ZEROSUM_SRC = ../zeroSum/src

OBJ  = main.o \
	   csv_read_write.o \
	   fusionKernel.o \
	   vectorizableKernelsAVX2.o \
	   vectorizableKernelsAVX512.o \
	   zeroSumApproximation.o \
	   zeroSumConstructors.o \
	   zeroSumCoordinateDescent.o \
	   zeroSumCoordinateDescentParallel.o \
	   zeroSumCostFunction.o \
	   zeroSumCV.o \
	   zeroSumDebugFunctions.o \
	   zeroSumLocalSearch.o \
	   zeroSumMoves.o \
	   zeroSumMultinomialParameterAmbiguity.o \
	   zeroSumSimulatedAnnealing.o \
	   zeroSumUtils.o

all: $(OBJ)
	$(MPICC) -flto $^ $(LIBS) -o zeroSum

gprof: clean gprofMain $(OBJ)
	$(MPICC) $(OBJ) $(LIBS) -g -lprofiler -o zeroSum

vectorizableKernelsAVX2.o: ../zeroSum/src/vectorizableKernelsAVX2.cpp
	$(CXX) $(CXXFLAGS) $(WARNINGS) -mavx2 -mfma -c $< -o $@

vectorizableKernelsAVX512.o:../zeroSum/src/vectorizableKernelsAVX512.cpp
	$(CXX) $(CXXFLAGS) $(WARNINGS) -mavx512f -c $< -o $@

gprofMain: main.cpp
	$(eval CXXFLAGS = -std=c++11 -O3 -g -march=$(ARCH) -mtune=$(ARCH) -DBUILD_WITH_AVX512 -DBUILD_WITH_AVX2)
	$(MPICC) $(CXXFLAGS) $(WARNINGS) -DG_PROF -c main.cpp

main.o: main.cpp
	$(MPICC) $(CXXFLAGS) $(WARNINGS) -c main.cpp

csv_read_write.o: csv_read_write.cpp csv_read_write.h
	$(MPICC) $(CXXFLAGS) $(WARNINGS) -c csv_read_write.cpp

%.o: $(ZEROSUM_SRC)/%.cpp
	$(CC) $(CXXFLAGS) $(WARNINGS) -c -o $@ $<


.PHONY: clean
clean:
	rm -f *.o
	rm -f zeroSum
	rm -f *.gch
