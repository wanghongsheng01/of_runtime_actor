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
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/graph/id_serialization.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/graph/normal_forward_compute_task_node.h"

namespace oneflow {

namespace {

bool IsLbiOnTaskEdge(const TaskEdge* edge, const LogicalBlobId& lbi) {
  for (const auto& regst_desc : edge->GetRegsts()) {
    if (regst_desc->HasLbi(lbi)) { return true; }
  }
  return false;
}

void GenSortedCompTaskNodes(const OpNode* op_node, std::vector<CompTaskNode*>* sorted_comp_tasks) {
  {
    {
      CompTaskNode* comp_task_node = NewCompTaskNode4OpNode(op_node);
      comp_task_node->set_machine_id(0);

      int32_t stream_index = 0;
      CHECK(!op_node->op().op_conf().has_stream_index_hint());

      comp_task_node->set_thrd_id(stream_index);
      comp_task_node->set_op_node(op_node);
      sorted_comp_tasks->push_back(comp_task_node);
    }
  }
}

BldSubTskGphMthd GetMthdForBldSubTskGph(const OpEdge* op_edge) {
  return &TaskGraph::BldSubTskGphByOneToOne;
}

}  // namespace

TaskGraph::TaskGraph() {
  OpGraph* op_graph = Global<OpGraph>::Get();
  HashMap<const OpNode*, std::vector<CompTaskNode*>> op_node2sorted_comp_tasks;

  op_graph->ForEachNode([&](const OpNode* op_node) {
    std::vector<CompTaskNode*>* sorted_comp_tasks = &(op_node2sorted_comp_tasks[op_node]);
    GenSortedCompTaskNodes(op_node, sorted_comp_tasks);
    for (CompTaskNode* comp_task : *sorted_comp_tasks) { AddAllocatedNode(comp_task); }
  });

  op_graph->ForEachEdge([&](const OpEdge* op_edge) {
    BldSubTskGphMthd method = GetMthdForBldSubTskGph(op_edge);
    (this->*method)(op_edge, op_node2sorted_comp_tasks.at(op_edge->src_node()),
                    op_node2sorted_comp_tasks.at(op_edge->dst_node()));
  });

}

TaskGraph::~TaskGraph() = default;

TaskNode* TaskGraph::GetProxyNode(TaskNode* src_node, const LogicalBlobId& lbi,
                                  int64_t dst_machine_id, int64_t dst_mem_zone_id) {
  int64_t src_mem_zone_id = src_node->MemZoneId121();
  const ProxyKey key(src_node, lbi, dst_machine_id, dst_mem_zone_id);
  if (proxy2node.find(key) != proxy2node.cend()) {
    return proxy2node.at(key);
  }
  CHECK(dst_mem_zone_id == src_mem_zone_id);
  proxy2node[key] = src_node;
  return src_node;
}

TaskNode* TaskGraph::GetProxyNode(TaskNode* src_node, const LogicalBlobId& lbi,
                                  const ParallelDesc& dst_parallel_desc, int64_t dst_parallel_id) {
  const int64_t dst_machine_id = 0;
  const int64_t dst_mem_zone_id = 0;
  return GetProxyNode(src_node, lbi, dst_machine_id, dst_mem_zone_id);
}

#define DEFINE_BLD_SUB_TASK_GRAPH_METHOD(method_name) \
  void TaskGraph::method_name BLD_SUB_TSK_GPH_MTHD_ARGS()

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByOneToOne) {
  CHECK_EQ(sorted_src_comp_tasks.size(), sorted_dst_comp_tasks.size());
  FOR_RANGE(size_t, i, 0, sorted_src_comp_tasks.size()) {
    for (const LogicalBlobId& lbi : op_edge->lbis()) {
      BuildTaskPath(sorted_src_comp_tasks.at(i), sorted_dst_comp_tasks.at(i), lbi);
    }
  }
}

void TaskGraph::ConnectWithLbi(TaskNode* src_node, TaskNode* dst_node, const LogicalBlobId& lbi) {
  if (src_node == dst_node) { return; }
  for (TaskEdge* out_edge : src_node->out_edges()) {
    TaskNode* out_node = out_edge->dst_node();
    if (out_node == dst_node) {
      out_edge->AddLbi(lbi);
      return;
    }
  }

  TaskEdge* connected_edge = NewEdge();
  connected_edge->AddLbi(lbi);
  Connect<TaskNode>(src_node, connected_edge, dst_node);
}

void TaskGraph::BuildTaskPath(TaskNode* src_node, TaskNode* dst_node, const LogicalBlobId& lbi) {
  int64_t dst_machine_id = dst_node->machine_id();
  int64_t dst_mem_zone_id = dst_node->MemZoneId121();
  TaskNode* proxy_node = GetProxyNode(src_node, lbi, dst_machine_id, dst_mem_zone_id);
  ConnectWithLbi(proxy_node, dst_node, lbi);
}

}  // namespace oneflow
