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
#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/common/id_util.h"
#include "oneflow/core/graph/id_serialization.h"
#include "oneflow/core/job/id_manager.h"

namespace oneflow {

namespace {

void ForEachDataEdge(const std::unordered_set<TaskEdge*>& edges,
                     const std::function<void(TaskEdge*)>& Handler) {
  for (TaskEdge* edge : edges) {
    const auto& regsts = edge->GetRegsts();
    int32_t data_regst_size =
        std::count_if(regsts.begin(), regsts.end(), [](const std::shared_ptr<RegstDesc>& regst) {
          return regst->regst_desc_type().has_data_regst_desc();
        });
    if (data_regst_size == regsts.size()) {
      Handler(edge);
    } else {
      CHECK_EQ(data_regst_size, 0);
    }
  }
}

}  // namespace

TaskNode::TaskNode()
    : machine_id_(-1), thrd_id_(-1), task_id_(-1) {}

std::shared_ptr<RegstDesc> TaskNode::GetProducedRegst(const std::string& name) {
  auto produced_regsts_it = produced_regsts_.find(name);
  if (produced_regsts_it == produced_regsts_.end()) {
    return nullptr;
  } else {
    return produced_regsts_it->second;
  }
}

const std::list<std::shared_ptr<RegstDesc>>& TaskNode::GetConsumedRegst(const std::string& name) {
  return consumed_regsts_.at(name);
}

std::shared_ptr<RegstDesc> TaskNode::GetSoleConsumedRegst(const std::string& name) {
  auto it = consumed_regsts_.find(name);
  if (it == consumed_regsts_.end()) { return nullptr; }
  const std::list<std::shared_ptr<RegstDesc>>& vec = it->second;
  CHECK_EQ(vec.size(), 1);
  return vec.front();
}

DeviceType TaskNode::device_type() const {
  return DeviceType::kCPU;
}

void TaskNode::set_machine_id(int64_t val) {
  CHECK_EQ(machine_id_, -1);
  machine_id_ = val;
  if (thrd_id_ != -1) { UpdateTaskId(); }
}

void TaskNode::set_thrd_id(int64_t val) {
  CHECK_EQ(thrd_id_, -1);
  thrd_id_ = val;
  CHECK_GE(thrd_id_, 0);
  if (machine_id_ != -1) { UpdateTaskId(); }
}

void TaskNode::ForEachConsumedDataRegst(
    const std::function<void(const std::string&, const RegstDesc*)>& Handler) const {
  for (const auto& pair : consumed_regsts_) {
    for (const auto& regst : pair.second) {
      if (!regst->regst_desc_type().has_data_regst_desc()) { continue; }
      Handler(pair.first, regst.get());
    }
  }
}

void TaskNode::ForEachProducedDataRegst(
    const std::function<void(const std::string&, RegstDesc*)>& Handler) {
  for (auto& pair : produced_regsts_) {
    if (!pair.second->regst_desc_type().has_data_regst_desc()) { continue; }
    Handler(pair.first, pair.second.get());
  }
}

void TaskNode::Build() { BuildExecGphAndRegst(); }

std::string TaskNode::VisualStr() const {
  std::stringstream ss;
  ss << TaskType_Name(GetTaskType()) << "\\n"
     << machine_id_ << ":" << thrd_id_ << "\\n"
     << task_id_;
  return ss.str();
}

bool TaskNode::IsMeaningLess() { return produced_regsts_.empty() && consumed_regsts_.empty(); }

void TaskNode::ToProto(TaskProto* task_proto) const {
  // Step1: process some scalar items.
  task_proto->set_task_type(GetTaskType());
  task_proto->set_machine_id(machine_id_);
  task_proto->set_thrd_id(thrd_id_);
  task_proto->set_task_id(task_id_);
  task_proto->set_job_id(GlobalJobDesc().job_id());

  // Step2: process exec_gph.
  exec_gph_.ToExecSequence(parallel_ctx(), task_proto->mutable_exec_sequence());

  // Step3: process produced_regst.
  auto* produced_regst_proto = task_proto->mutable_produced_regst_desc();
  for (auto& pair : produced_regsts_) {
    RegstDescProto regst_desc_proto;
    pair.second->ToProto(&regst_desc_proto);
    CHECK(produced_regst_proto->insert({pair.first, regst_desc_proto}).second);
  }

  // Step4: process consumed_regst.
  auto* consumed_regst_proto = task_proto->mutable_consumed_regst_desc_id();
  for (const auto& pair : consumed_regsts_) {
    RegstDescIdSet regst_desc_ids;
    for (const std::shared_ptr<RegstDesc>& regst : pair.second) {
      regst_desc_ids.add_regst_desc_id(regst->regst_desc_id());
    }
    CHECK(consumed_regst_proto->insert({pair.first, regst_desc_ids}).second);
  }
}

int64_t TaskNode::MemZoneId121() const {
  CHECK(device_type() == DeviceType::kCPU);
  return 0;
}

void TaskNode::BindEdgeWithProducedRegst(TaskEdge* edge, const std::string& name) {
  edge->AddRegst(name, GetProducedRegst(name));
}

std::shared_ptr<RegstDesc> TaskNode::ProduceRegst(const std::string& name, bool enable_reuse_mem) {
  return ProduceRegst(name, enable_reuse_mem, 1, kMaxRegisterNum);
}

std::shared_ptr<RegstDesc> TaskNode::ProduceRegst(const std::string& name, bool enable_reuse_mem,
                                                  int32_t min_register_num,
                                                  int32_t max_register_num) {
  RegstDescTypeProto regst_desc_type;
  regst_desc_type.mutable_data_regst_desc();
  return ProduceRegst(name, enable_reuse_mem, min_register_num, max_register_num, regst_desc_type);
}

std::shared_ptr<RegstDesc> TaskNode::ProduceRegst(const std::string& name, bool enable_reuse_mem,
                                                  int32_t min_register_num,
                                                  int32_t max_register_num,
                                                  const RegstDescTypeProto& regst_desc_type) {
  auto regst =
      NewProducedRegst(enable_reuse_mem, min_register_num, max_register_num, regst_desc_type);
  CHECK(produced_regsts_.emplace(name, regst).second);
  return regst;
}

std::shared_ptr<RegstDesc> TaskNode::NewProducedRegst(bool enable_reuse_mem,
                                                      int32_t min_register_num,
                                                      int32_t max_register_num,
                                                      const RegstDescTypeProto& regst_desc_type) {
  auto regst = std::make_shared<RegstDesc>();
  regst->set_producer(this);
  *(regst->mut_regst_desc_type()) = regst_desc_type;
  regst->set_enable_reuse_mem(GlobalJobDesc().enable_reuse_mem() && enable_reuse_mem);
  InitProducedRegstMemCase(regst.get());
  return regst;
}

void TaskNode::InitProducedRegstMemCase(RegstDesc* regst) {
  InitProducedRegstMemCase(regst->mut_mem_case());
}

void TaskNode::InitProducedRegstMemCase(MemoryCase* mem_case) {
  CHECK(device_type() == DeviceType::kCPU);
  mem_case->mutable_host_mem();
}

void TaskNode::ConsumeRegst(const std::string& name) {
  consumed_regsts_.emplace(name, std::list<std::shared_ptr<RegstDesc>>{});
}

void TaskNode::ConsumeRegst(const std::string& name, const std::shared_ptr<RegstDesc>& regst) {
  regst->AddConsumer(this);
  consumed_regsts_[name].push_back(regst);
}

void TaskNode::UpdateTaskId() {
  CHECK_NE(machine_id_, -1);
  CHECK_NE(thrd_id_, -1);
  static int task_idx = 0;
  TaskId task_id{thrd_id_, task_idx++};
  task_id_ = SerializeTaskIdToInt64(task_id);
}

int64_t TaskNode::GlobalWorkStreamId() const {
  CHECK_NE(task_id_, -1);
  return Global<IDMgr>::Get()->GlobalWorkStreamId4TaskId(task_id_);
}

std::shared_ptr<RegstDesc> TaskEdge::GetRegst(const std::string& name_in_producer) const {
  return name_in_producer2regst_.at(name_in_producer);
}

std::shared_ptr<RegstDesc> TaskEdge::GetSoleRegst() const {
  CHECK_EQ(name_in_producer2regst_.size(), 1);
  return name_in_producer2regst_.begin()->second;
}

std::vector<std::shared_ptr<RegstDesc>> TaskEdge::GetRegsts() const {
  std::vector<std::shared_ptr<RegstDesc>> regst_descs;
  for (auto& pair : name_in_producer2regst_) { regst_descs.emplace_back(pair.second); }
  return regst_descs;
}

void TaskEdge::AddRegst(const std::string& name_in_producer,
                        const std::shared_ptr<RegstDesc>& regst) {
  CHECK(name_in_producer2regst_.emplace(name_in_producer, regst).second);
}

void TaskEdge::CheckRegstLbiValid() const {
  HashMap<LogicalBlobId, std::shared_ptr<RegstDesc>> lbi2data_regst;
  for (auto& pair : name_in_producer2regst_) {
    std::shared_ptr<RegstDesc> regst = pair.second;
    if (regst->regst_desc_type().has_data_regst_desc()) {
      // NOTE(chengcheng): regst_desc_type is Set, BUT regst_desc_type.data_regst_desc is UNSET!
      //  So you can ONLY use NumOfLbi and ForEachLbi interface.
      CHECK_EQ(regst->NumOfLbi(), 1);
      regst->ForEachLbi(
          [&](const LogicalBlobId& lbi) { CHECK(lbi2data_regst.emplace(lbi, regst).second); });
    }
  }

  CHECK_EQ(lbi2data_regst.size(), lbis_.size())
      << " \n\n TaskEdge lbi and regst NOT match."
      << " TaskEdge: edge_id = " << edge_id() << " From: [" << src_node()->VisualStr() << "] To: ["
      << dst_node()->VisualStr() << "]\n";
  for (auto& lbi : lbis_) {
    CHECK(lbi2data_regst.find(lbi) != lbi2data_regst.end())
        << " \n\n Cannot find lbi: " << lbi.DebugString() << " in TaskEdge From: ["
        << src_node()->VisualStr() << "] To: [" << dst_node()->VisualStr() << "]\n\n";
  }
}

void TaskNode::ForEachInDataEdge(const std::function<void(TaskEdge*)>& Handler) const {
  ForEachDataEdge(in_edges(), Handler);
}

void TaskNode::ForEachOutDataEdge(const std::function<void(TaskEdge*)>& Handler) const {
  ForEachDataEdge(out_edges(), Handler);
}

void TaskNode::ForEachNodeOnInDataEdge(const std::function<void(TaskNode*)>& Handler) const {
  ForEachInDataEdge([&](TaskEdge* in_edge) { Handler(in_edge->src_node()); });
}

void TaskNode::ForEachNodeOnOutDataEdge(const std::function<void(TaskNode*)>& Handler) const {
  ForEachOutDataEdge([&](TaskEdge* out_edge) { Handler(out_edge->dst_node()); });
}

void TaskNode::ForEachNodeOnInOutDataEdge(const std::function<void(TaskNode*)>& Handler) const {
  ForEachNodeOnInDataEdge(Handler);
  ForEachNodeOnOutDataEdge(Handler);
}

TaskEdge* TaskNode::GetSoleEdge(void (TaskNode::*ForEachEdge)(const std::function<void(TaskEdge*)>&)
                                    const) const {
  TaskEdge* ret = nullptr;
  (this->*ForEachEdge)([&](TaskEdge* edge) {
    CHECK(ret == nullptr);
    ret = edge;
  });
  CHECK_NOTNULL(ret);
  return ret;
}

size_t TaskNode::GetEdgesSize(void (TaskNode::*ForEachEdge)(const std::function<void(TaskEdge*)>&)
                                  const) const {
  size_t size = 0;
  (this->*ForEachEdge)([&](TaskEdge* edge) { ++size; });
  return size;
}

}  // namespace oneflow
