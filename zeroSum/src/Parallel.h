#ifndef PARALLEL_H
#define PARALLEL_H

#include <cmath>
#include <functional>
#include <future>
#include <vector>

class Parallel {
   public:
    Parallel(size_t);
    ~Parallel();

    template <class F, class... Args>
    void doParallel(size_t numberTasks, F&& f, Args&&... args);
    template <class F, class... Args>
    void doParallelChunked(size_t numberTasks, F&& f, Args&&... args);
    size_t maxThreads;

   private:
    std::vector<size_t> batchSizes;
    std::vector<size_t> start;
};

inline Parallel::Parallel(size_t maxThreads) : maxThreads(maxThreads) {
    // 1 thread should imply that the work is executed by the main thread
    if (this->maxThreads == 0)
        this->maxThreads = 1;

    size_t availableCores = std::thread::hardware_concurrency();
    if (this->maxThreads > availableCores)
        this->maxThreads = availableCores;

    batchSizes.resize(this->maxThreads);
    start.resize(this->maxThreads);
}

inline Parallel::~Parallel() {}

// openmp like parallelFor
template <class F, class... Args>
inline void Parallel::doParallel(size_t numberTasks, F&& f, Args&&... args) {
    std::vector<std::future<void>> results;
    size_t todoMaster = floor((double)numberTasks / (double)maxThreads);

    // distribute work to threadpool
    for (size_t i = todoMaster; i < numberTasks; i++) {
        results.emplace_back(
            std::async(std::launch::async,
                       std::bind([&](size_t i) { f(i, args...); }, i)));
    }

    // do the remaining work
    for (size_t i = 0; i < todoMaster; i++) {
        f(i, args...);
    }

    // get results
    for (auto&& result : results)
        result.get();
}

// openmp like parallelFor
template <class F, class... Args>
inline void Parallel::doParallelChunked(size_t numberTasks,
                                        F&& f,
                                        Args&&... args) {
    size_t jobThreads = maxThreads;
    if (numberTasks < jobThreads)
        jobThreads = numberTasks;

    size_t size = numberTasks / jobThreads;
    size_t remaining = numberTasks % jobThreads;

    for (size_t i = 0; i < jobThreads; i++) {
        batchSizes[i] = size;
        if (i < remaining)
            batchSizes[i] += 1;
    }

    start[0] = 0;
    for (size_t i = 1; i < jobThreads; i++)
        start[i] = start[i - 1] + batchSizes[i - 1];

    std::vector<std::future<void>> results;
    for (size_t batch = 1; batch < jobThreads; batch++) {
        results.emplace_back(
            std::async(std::launch::async,
                       std::bind(
                           [&](size_t batch) {
                               for (size_t i = start[batch];
                                    i < start[batch] + batchSizes[batch]; i++) {
                                   f(i, args...);
                               }
                           },
                           batch)));
    }

    // do the remaining work
    for (size_t i = 0; i < batchSizes[0]; i++) {
        f(i, args...);
    }

    // get results
    for (auto&& result : results)
        result.get();
}

#endif
