#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <cmath>
#include <functional>
#include <future>
#include <queue>

class ThreadPool {
   public:
    ThreadPool(size_t);
    ~ThreadPool();

    template <class F, class... Args>
    void doParallel(size_t numberTasks, F&& f, Args&&... args);
    template <class F, class... Args>
    void doParallelChunked(size_t numberTasks, F&& f, Args&&... args);

   private:
    std::vector<std::thread> threadpool;
    std::vector<size_t> batchSizes;
    std::vector<size_t> start;
    size_t maxThreads;
    std::queue<std::packaged_task<void()>> jobs;

    std::mutex mutex;
    std::condition_variable sync;

    bool stop;
};

inline ThreadPool::ThreadPool(size_t maxThreads)
    : maxThreads(maxThreads), stop(false) {
    // 1 thread should imply that the work is executed by the main thread
    if (this->maxThreads == 0)
        this->maxThreads = 1;

    size_t availableCores = std::thread::hardware_concurrency();
    if (this->maxThreads > availableCores)
        this->maxThreads = availableCores;

    batchSizes.resize(this->maxThreads);
    start.resize(this->maxThreads);
    threadpool.reserve(this->maxThreads);

    for (size_t i = 1; i < this->maxThreads; ++i) {
        auto endlessLoop = [this]() {
            while (true) {
                std::packaged_task<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->mutex);

                    this->sync.wait(lock, [this] {
                        return !this->jobs.empty() || this->stop;
                    });

                    if (this->stop)
                        return;
                    task = std::move(this->jobs.front());
                    this->jobs.pop();
                }
                task();
            }
        };
        threadpool.push_back(std::thread(endlessLoop));
    }
}

inline ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(mutex);
        stop = true;
    }
    sync.notify_all();
    for (auto& thread : threadpool)
        thread.join();
}

// openmp like parallelFor
template <class F, class... Args>
inline void ThreadPool::doParallel(size_t N, F&& f, Args&&... args) {
    std::vector<std::future<void>> results;
    size_t todoMaster = floor((double)N / (double)maxThreads);

    // distribute work to threadpool
    for (size_t i = todoMaster; i < N; i++) {
        std::packaged_task<void()> task(
            std::bind(std::forward<F>(f), i, std::forward<Args>(args)...));

        std::future<void> res = task.get_future();
        {
            std::unique_lock<std::mutex> lock(mutex);
            jobs.emplace(std::move(task));
        }
        sync.notify_one();
        results.emplace_back(std::move(res));
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
inline void ThreadPool::doParallelChunked(size_t numberTasks,
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
        std::packaged_task<void()> task(std::bind(
            [&](size_t batch) {
                for (size_t i = start[batch];
                     i < start[batch] + batchSizes[batch]; i++) {
                    f(i, args...);
                }
            },
            batch));
        std::future<void> res = task.get_future();
        {
            std::unique_lock<std::mutex> lock(mutex);
            jobs.emplace(std::move(task));
        }
        results.emplace_back(std::move(res));
    }
    sync.notify_all();
    // do the remaining work
    for (size_t i = 0; i < batchSizes[0]; i++) {
        f(i, args...);
    }

    // get results
    for (auto&& result : results)
        result.get();
}
#endif