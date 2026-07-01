// Copyright 2024 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "threadpool.h"

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <absl/base/attributes.h>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace mujoco::python {

ABSL_CONST_INIT thread_local int ThreadPool::worker_id_ = -1;

namespace {

#if defined(__linux__)
// Pin the target pthread to a single CPU. Returns 0 on success, otherwise
// the errno from pthread_setaffinity_np.
int PinPthreadToCpu(pthread_t thread, int cpu_id) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu_id, &cpuset);
  return pthread_setaffinity_np(thread, sizeof(cpuset), &cpuset);
}

// Read the affinity mask of the target pthread into a sorted CPU id list.
// Returns an empty vector on syscall failure.
std::vector<int> ReadPthreadAffinity(pthread_t thread) {
  cpu_set_t observed;
  CPU_ZERO(&observed);
  if (pthread_getaffinity_np(thread, sizeof(observed), &observed) != 0) {
    return {};
  }
  std::vector<int> out;
  for (int c = 0; c < CPU_SETSIZE; ++c) {
    if (CPU_ISSET(c, &observed)) out.push_back(c);
  }
  return out;
}
#endif

}  // namespace

// ThreadPool constructor
ThreadPool::ThreadPool(int num_threads) : ctr_(0) {
  for (int i = 0; i < num_threads; i++) {
    threads_.push_back(std::thread(&ThreadPool::WorkerThread, this, i));
  }
}

ThreadPool::ThreadPool(int num_threads, std::vector<int> worker_cpu_ids)
    : ctr_(0) {
  const bool want_pin = !worker_cpu_ids.empty();
  if (want_pin &&
      static_cast<int>(worker_cpu_ids.size()) != num_threads) {
    // A mismatched cpu_ids length is a caller bug, not a soft fallback:
    // silently ignoring it would mask configuration mistakes and violate
    // the "worker fixed on requested CPU" acceptance criterion.
    throw std::runtime_error(
        "ThreadPool: worker_cpu_ids size (" +
        std::to_string(worker_cpu_ids.size()) +
        ") must equal num_threads (" + std::to_string(num_threads) + ")");
  }

  // Launch all workers first. Affinity is applied from this thread via
  // native_handle() so we can observe the syscall's return value and fail
  // loud when the kernel rejects the request.
  for (int i = 0; i < num_threads; i++) {
    threads_.push_back(std::thread(&ThreadPool::WorkerThread, this, i));
  }

  if (!want_pin) return;

#if defined(__linux__)
  worker_affinities_.resize(num_threads);
  for (int i = 0; i < num_threads; i++) {
    const int cpu = worker_cpu_ids[i];
    int rc = PinPthreadToCpu(threads_[i].native_handle(), cpu);
    if (rc != 0) {
      // Tear down cleanly so the destructor is not needed. Push shutdown
      // sentinels for every worker we already launched, then join.
      {
        std::unique_lock<std::mutex> lock(m_);
        for (std::size_t j = 0; j < threads_.size(); j++) {
          queue_.push(nullptr);
        }
        cv_in_.notify_all();
      }
      for (auto& t : threads_) {
        if (t.joinable()) t.join();
      }
      threads_.clear();
      worker_affinities_.clear();
      throw std::runtime_error(
          "ThreadPool: pthread_setaffinity_np failed for worker " +
          std::to_string(i) + " cpu " + std::to_string(cpu) +
          " (errno=" + std::to_string(rc) + ": " + std::strerror(rc) + ")");
    }
    worker_affinities_[i] =
        ReadPthreadAffinity(threads_[i].native_handle());
  }
#else
  // Non-Linux platforms have no affinity API. Leave worker_affinities_
  // empty so callers can detect the fallback via WorkerAffinities().
  (void)num_threads;
#endif
}

// ThreadPool destructor
ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(m_);
    for (int i = 0; i < threads_.size(); i++) {
      queue_.push(nullptr);
    }
    cv_in_.notify_all();
  }
  for (auto& thread : threads_) {
    thread.join();
  }
}

// ThreadPool scheduler
void ThreadPool::Schedule(std::function<void()> task) {
  std::unique_lock<std::mutex> lock(m_);
  queue_.push(std::move(task));
  cv_in_.notify_one();
}

// Run fn exactly once on each worker, blocking until all have completed.
void ThreadPool::RunOnEachWorker(const std::function<void(int)>& fn) {
  const int n = NumThreads();
  if (n == 0) return;

  // Barrier shared by the n init tasks. Lives on this (non-worker) stack
  // frame; safe because we block in WaitCount(n) below until every task has
  // completed, so no task outlives these locals.
  std::mutex barrier_mutex;
  std::condition_variable barrier_cv;
  int arrived = 0;

  ResetCount();
  for (int i = 0; i < n; ++i) {
    Schedule([&]() {
      fn(WorkerId());
      // Do not let this task return until all n workers have entered the
      // barrier. Because a worker only returns to the dequeue loop after its
      // task returns, this guarantees every worker pops exactly one init task
      // (a fast worker cannot grab a second and starve a slower one).
      std::unique_lock<std::mutex> lock(barrier_mutex);
      if (++arrived == n) {
        barrier_cv.notify_all();
      } else {
        barrier_cv.wait(lock, [&]() { return arrived == n; });
      }
    });
  }
  WaitCount(n);
}

// ThreadPool worker
void ThreadPool::WorkerThread(int i) {
  worker_id_ = i;
  while (true) {
    auto task = [&]() {
      std::unique_lock<std::mutex> lock(m_);
      cv_in_.wait(lock, [&]() { return !queue_.empty(); });
      std::function<void()> task = std::move(queue_.front());
      queue_.pop();
      cv_in_.notify_one();
      return task;
    }();
    if (task == nullptr) {
      {
        std::unique_lock<std::mutex> lock(m_);
        ++ctr_;
        cv_ext_.notify_one();
      }
      break;
    }
    task();

    {
      std::unique_lock<std::mutex> lock(m_);
      ++ctr_;
      cv_ext_.notify_one();
    }
  }
}

}  // namespace mujoco::python
