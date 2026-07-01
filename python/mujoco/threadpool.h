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

#ifndef MUJOCO_PYTHON_THREADPOOL_H_
#define MUJOCO_PYTHON_THREADPOOL_H_

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <absl/base/attributes.h>

namespace mujoco::python {

// ThreadPool class
class ThreadPool {
 public:
  // constructor
  explicit ThreadPool(int num_threads);

  // Constructor with optional per-worker CPU affinity. If worker_cpu_ids is
  // non-empty, its size must equal num_threads (mismatch throws). On Linux
  // each worker is pinned to its requested CPU via pthread_setaffinity_np;
  // if the syscall fails the constructor throws std::runtime_error after
  // joining any already-launched workers. Phase 1 acceptance requires
  // "worker fixed on requested CPU" as a hard invariant, so silent
  // fallback is only allowed on platforms without an affinity API (in
  // that case WorkerAffinities() returns an empty vector so callers can
  // detect the fallback).
  ThreadPool(int num_threads, std::vector<int> worker_cpu_ids);

  // destructor
  ~ThreadPool();

  int NumThreads() const { return threads_.size(); }

  // Per-worker observed CPU affinity mask read back after pinning. Empty
  // outer vector means "no affinity information available" (either
  // pinning was not requested, or the platform has no affinity API).
  // On Linux, entry i is the set of CPU ids that pthread_getaffinity_np
  // reported for worker i. Tests can assert
  // WorkerAffinities()[i] == {worker_cpu_ids[i]}.
  const std::vector<std::vector<int>>& WorkerAffinities() const {
    return worker_affinities_;
  }

  // returns an ID between 0 and NumThreads() - 1. must be called within
  // worker thread (returns -1 if not).
  static int WorkerId() { return worker_id_; }

  // ----- methods ----- //
  // set task for threadpool
  void Schedule(std::function<void()> task);

  // return number of tasks completed
  std::uint64_t GetCount() { return ctr_; }

  // reset count to zero
  void ResetCount() { ctr_ = 0; }

  // wait for count, then return
  void WaitCount(int value) {
    std::unique_lock<std::mutex> lock(m_);
    cv_ext_.wait(lock, [&]() { return this->GetCount() >= value; });
  }

 private:
  // ----- methods ----- //

  // execute task with available thread
  void WorkerThread(int i);

  ABSL_CONST_INIT static thread_local int worker_id_;

  // ----- members ----- //
  std::vector<std::thread> threads_;
  // Observed CPU mask per worker after pinning. Outer size 0 means "no
  // pinning attempted or platform has no affinity API".
  std::vector<std::vector<int>> worker_affinities_;
  std::mutex m_;
  std::condition_variable cv_in_;
  std::condition_variable cv_ext_;
  std::queue<std::function<void()>> queue_;
  std::uint64_t ctr_;
};

}  // namespace mujoco::python

#endif  // MUJOCO_PYTHON_THREADPOOL_H_
