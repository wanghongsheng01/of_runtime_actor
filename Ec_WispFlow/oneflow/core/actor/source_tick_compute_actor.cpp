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
#include "oneflow/core/actor/source_tick_compute_actor.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/record/record.pb.h"

namespace oneflow {

void SourceTickComputeActor::VirtualCompActorInit(const TaskProto& task_proto) {
  OF_SET_MSG_HANDLER(&SourceTickComputeActor::HandlerWaitToStart);
}

void SourceTickComputeActor::Act() {
  LOG(INFO) << __func__;
  AsyncLaunchKernel([&](int64_t regst_desc_id) -> Regst* { return nullptr; });
}

int SourceTickComputeActor::HandlerWaitToStart(const ActorMsg& msg) {
  CHECK_EQ(msg.actor_cmd(), ActorCmd::kStart);
  OF_SET_MSG_HANDLER(&SourceTickComputeActor::HandlerNormal);
  return ProcessMsg(msg);
}

REGISTER_ACTOR(kSourceTick, SourceTickComputeActor);

}  // namespace oneflow
