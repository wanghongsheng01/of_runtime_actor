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
#include <climits>
#include "oneflow/core/graph/id_serialization.h"

namespace oneflow {

// TaskId encode (may be extended to 128 bit in future)
// |                               StreamId                                  | task_index |
// | -------------------------------- 43 ----------------------------------- | --- 21 --- |
// |                                      TaskId                                          |
// | ----------------------------------- 64 bit ----------------------------------------- |

namespace {

constexpr size_t kInt64Bits = sizeof(int64_t) * CHAR_BIT;

}  // namespace

namespace task_id_const {

constexpr size_t kStreamIndexShift = TaskId::kTaskIndexBits;
static_assert(kInt64Bits == kStreamIndexShift + TaskId::kStreamIndexBits, "");

constexpr int64_t kTaskIndexInt64Mask = (int64_t{1} << TaskId::kTaskIndexBits) - 1;
constexpr int64_t kStreamIndexInt64Mask = ((int64_t{1} << TaskId::kStreamIndexBits) - 1)
                                          << kStreamIndexShift;

}  // namespace task_id_const

int64_t SerializeTaskIdToInt64(const TaskId& task_id) {
  int64_t id = static_cast<int64_t>(task_id.task_index());
  id |= static_cast<int64_t>(task_id.stream_index())
        << task_id_const::kStreamIndexShift;
  return id;
}

TaskId DeserializeTaskIdFromInt64(int64_t task_id_val) {
  int64_t stream_index =
      (task_id_val & task_id_const::kStreamIndexInt64Mask) >> task_id_const::kStreamIndexShift;
  int64_t task_index = task_id_val & task_id_const::kTaskIndexInt64Mask;
  return TaskId{static_cast<int64_t>(stream_index), static_cast<int64_t>(task_index)};
}

}  // namespace oneflow
