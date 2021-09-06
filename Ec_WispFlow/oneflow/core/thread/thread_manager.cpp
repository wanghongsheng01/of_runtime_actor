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
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/thread/cpu_thread.h"
#include "oneflow/core/common/id_util.h"
#include "oneflow/core/graph/id_serialization.h"

namespace oneflow {

ThreadMgr::~ThreadMgr() {
  for (auto& thread_pair : threads_) {
    ActorMsg msg = ActorMsg::BuildCommandMsg(-1, ActorCmd::kStopThread);
    thread_pair.second->GetMsgChannelPtr()->Send(msg);
    thread_pair.second.reset();
    LOG(INFO) << "actor thread " << thread_pair.first << " finish";
  }
}

Thread* ThreadMgr::GetThrd(int64_t thrd_id) {
  auto iter = threads_.find(thrd_id);
  CHECK(iter != threads_.end()) << "thread " << thrd_id << " not found";
  return iter->second.get();
}

ThreadMgr::ThreadMgr(const Plan& plan) {
  for (const TaskProto& task : plan.task()) {
    TaskId task_id = DeserializeTaskIdFromInt64(task.task_id());
    int64_t stream_index = task_id.stream_index();
    int64_t thrd_id = stream_index;
    CHECK_EQ(task.thrd_id(), thrd_id);
    if (threads_.find(thrd_id) != threads_.end()) { continue; }
    threads_[thrd_id].reset(new CpuThread(thrd_id));
  }
}

}  // namespace oneflow
