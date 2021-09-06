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
#include "oneflow/core/job/compiler.h"
#include "oneflow/core/job/plan_util.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/thread/thread_pool.h"
#include "oneflow/core/common/blocking_counter.h"

namespace oneflow {

void Compiler::Compile(Job* job, Plan* plan, bool need_job_complete) const {
  // Step2: new Global<OpGraph> and set log configs.
  Global<OpGraph>::New(*job);
  const JobDesc& job_desc = GlobalJobDesc();

  // Step3: build task_gph.
  // TODO(levi): we can rewrite this part of code in visitor pattern.
  auto task_gph = std::make_unique<TaskGraph>();
  using std::placeholders::_1;
  task_gph->ForEachNode(std::bind(&TaskNode::ProduceAllRegstsAndBindEdges, _1));
  task_gph->ForEachNode(std::bind(&TaskNode::ConsumeAllRegsts, _1));
  task_gph->TopoForEachNode(&TaskNode::Build);
  task_gph->ForEachEdge([&](TaskEdge* task_edge) { task_edge->CheckRegstLbiValid(); });

  // Step4: put infomation from task_gph into plan.
  const int64_t node_num = task_gph->node_num();
  const int64_t cpu_num = std::thread::hardware_concurrency();
  const int64_t thread_pool_size = std::min(node_num, cpu_num);
  BlockingCounter counter(node_num);
  std::mutex mtx;
  ThreadPool thread_pool(thread_pool_size);
  task_gph->ForEachNode([&](TaskNode* task_node) {
    thread_pool.AddWork([task_node, plan, &job_desc, &counter, &mtx]() {
      if (!task_node->IsMeaningLess()) {
        TaskProto task_proto;
        task_node->ToProto(&task_proto);
        {
          std::unique_lock<std::mutex> guard(mtx);
          plan->mutable_task()->Add(std::move(task_proto));
        }  // guard(mtx)
      }
      counter.Decrease();
    } /* thread_pool.AddWork */);
  } /* task_gph->ForEachNode */);
  counter.WaitUntilCntEqualZero();
  // NOTE(levi): release task_gph here to decrise memory peak.
  task_gph.reset();

  // Step5: post-process for plan and delete Global<OpGraph>.
  PlanUtil::SetUniqueMemBlockId4UnreusedMemRegst(plan);
  PlanUtil::GenMemBlockAndChunk4Plan(plan);
  Global<OpGraph>::Delete();
}

}  // namespace oneflow
