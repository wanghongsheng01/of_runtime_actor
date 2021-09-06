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
#include "oneflow/core/job/plan_util.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/memory/memory_case_util.h"
#include "oneflow/core/register/runtime_register_desc.h"

namespace oneflow {

void PlanUtil::SetUniqueMemBlockId4UnreusedMemRegst(Plan* plan) {
  for (int i = 0; i < plan->task_size(); i++) {
    TaskProto* task = plan->mutable_task(i);
    for (auto& pair : *task->mutable_produced_regst_desc()) {
      RegstDescProto* regst_desc = &pair.second;
      if (regst_desc->mem_block_id() == -1) {
        CHECK_EQ(regst_desc->mem_block_offset(), -1);
        regst_desc->set_mem_block_id(Global<IDMgr>::Get()->NewMemBlockId());
        regst_desc->set_mem_block_offset(0);
      }
    }
  }
}

void PlanUtil::GenMemBlockAndChunk4Plan(Plan* plan) {
  HashMap<int64_t, MemBlockProto> mem_block_id2mem_block;
  // mzuid = memory zone unique id
  HashMap<int64_t, ChunkProto> mzuid2chunk;

  auto GenMemBlock4RegstIfNeed = [&](RegstDescProto* regst_desc, const TaskProto* task) {
    const int64_t job_id = task->job_id();
    const int64_t machine_id = task->machine_id();
    const int64_t thrd_id = task->thrd_id();
    int64_t mem_block_id = regst_desc->mem_block_id();
    int64_t mem_block_offset = regst_desc->mem_block_offset();
    CHECK_NE(mem_block_id, -1);
    CHECK_NE(mem_block_offset, -1);
    CHECK_EQ(regst_desc->separated_header_mem_block_id(), -1);

    RtRegstDesc rt_regst_desc(*regst_desc);
    int64_t regst_main_size = rt_regst_desc.TotalMainByteSize4AllRegst();

    if (mem_block_id2mem_block.find(mem_block_id) == mem_block_id2mem_block.end()) {
      MemBlockProto mem_block;
      mem_block.set_mem_block_id(mem_block_id);
      mem_block.add_job_id(job_id);
      mem_block.set_machine_id(machine_id);
      *(mem_block.mutable_mem_case()) = regst_desc->mem_case();
      mem_block.set_mem_size(regst_main_size + mem_block_offset);
      mem_block.set_thrd_id_hint(thrd_id);
      CHECK(mem_block_id2mem_block.emplace(mem_block.mem_block_id(), mem_block).second);
    } else {
      MemBlockProto* mem_block = &(mem_block_id2mem_block.at(mem_block_id));
      CHECK_EQ(mem_block->job_id(0), job_id);
      CHECK_EQ(mem_block->machine_id(), machine_id);
      CHECK(mem_block->mem_case() == regst_desc->mem_case());
      mem_block->set_mem_size(std::max(mem_block->mem_size(), regst_main_size + mem_block_offset));
    }
  };

  for (int i = 0; i < plan->task_size(); i++) {
    TaskProto* task = plan->mutable_task(i);
    for (auto& pair : *task->mutable_produced_regst_desc()) {
      GenMemBlock4RegstIfNeed(&pair.second, task);
    }
  }

  for (auto& pair : mem_block_id2mem_block) {
    MemBlockProto* mem_block = &pair.second;
    CHECK(mem_block->has_chunk_id() == false);
    CHECK(mem_block->has_chunk_offset() == false);
  }

  for (const auto& pair : mem_block_id2mem_block) {
    *(plan->mutable_block_chunk_list()->add_mem_block()) = pair.second;
  }

  for (const auto& pair : mzuid2chunk) {
    *(plan->mutable_block_chunk_list()->add_chunk()) = pair.second;
  }
}

}  // namespace oneflow
