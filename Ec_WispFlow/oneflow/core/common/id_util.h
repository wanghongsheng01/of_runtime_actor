/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_COMMON_ID_UTIL_H_
#define ONEFLOW_CORE_COMMON_ID_UTIL_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

// TaskId encode (may be extended to 128 bit in future)
// |                               StreamId                                  | task_index |
// | -------------------------------- 43 ----------------------------------- | --- 21 --- |
// |                                      TaskId                                          |
// | ----------------------------------- 64 bit ----------------------------------------- |
class TaskId final {
 public:
  constexpr static size_t kStreamIndexBits = 43;
  constexpr static size_t kTaskIndexBits = 21;
  TaskId(int64_t stream_index, int64_t task_index)
      : stream_index_(stream_index), task_index_(task_index) { }
  int64_t stream_index() const { return stream_index_; }
  int64_t task_index() const { return task_index_; }
  bool operator==(const TaskId& rhs) const {
    return stream_index_ == rhs.stream_index_ && task_index_ == rhs.task_index_;
  }
  bool operator!=(const TaskId& rhs) const { return !(*this == rhs); }
  size_t hash() const {
    size_t hash = std::hash<int64_t>{}(stream_index_);
    HashCombine(&hash, std::hash<int64_t>{}(task_index_));
    return hash;
  }

 private:
  int64_t stream_index_ {0};
  int64_t task_index_ {0};
};

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::TaskId> {
  size_t operator()(const oneflow::TaskId& task_id) const { return task_id.hash(); }
};

}  // namespace std

#endif  // ONEFLOW_CORE_COMMON_ID_UTIL_H_
