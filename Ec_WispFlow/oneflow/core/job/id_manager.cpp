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
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/common/id_util.h"
#include "oneflow/core/graph/id_serialization.h"

namespace oneflow {

int64_t IDMgr::ThrdId4ActorId(int64_t actor_id) const {
  return DeserializeTaskIdFromInt64(actor_id).stream_index();
}

int64_t IDMgr::GlobalWorkStreamId4TaskId(int64_t task_id) const {
  return DeserializeTaskIdFromInt64(task_id).stream_index();
}

int64_t IDMgr::GlobalWorkStreamId4ActorId(int64_t actor_id) const {
  return GlobalWorkStreamId4TaskId(actor_id);
}

int64_t IDMgr::GlobalThrdId4TaskId(int64_t task_id) const {
  return DeserializeTaskIdFromInt64(task_id).stream_index();
}

IDMgr::IDMgr() {
  cpu_device_num_ = 1;
  CHECK_LT(cpu_device_num_, (static_cast<int64_t>(1) << thread_id_bit_num_) - 3);
  regst_desc_id_count_ = 0;
  mem_block_id_count_ = 0;
  chunk_id_count_ = 0;
}

}  // namespace oneflow
