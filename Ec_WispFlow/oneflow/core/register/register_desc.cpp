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
#include "oneflow/core/register/register_desc.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/runtime_register_desc.h"

namespace oneflow {

RegstDesc::RegstDesc() {
  regst_desc_id_ = Global<IDMgr>::Get()->NewRegstDescId();
  producer_ = nullptr;
  mem_block_id_ = -1;
  mem_block_offset_ = -1;
}

int64_t RegstDesc::mem_block_offset() const {
  CHECK_GE(mem_block_offset_, 0);
  return mem_block_offset_;
}

void RegstDesc::AddConsumer(const TaskNode* new_consumer) {
  CHECK(consumers_.insert(new_consumer).second);
}

void RegstDesc::DeleteConsumer(const TaskNode* consumer) {
  CHECK_EQ(consumers_.erase(consumer), 1);
}

BlobDesc* RegstDesc::AddLbi(const LogicalBlobId& lbi) {
  CHECK(lbi2blob_desc_.find(lbi) == lbi2blob_desc_.end());
  BlobDesc* blob_desc = new BlobDesc(GlobalJobDesc().DefaultDataType());
  lbi2blob_desc_[lbi].reset(blob_desc);
  return blob_desc;
}

const BlobDesc* RegstDesc::GetBlobDesc(const LogicalBlobId& lbi) const {
  return const_cast<RegstDesc*>(this)->MutBlobDesc(lbi);
}

bool RegstDesc::HasLbi(const LogicalBlobId& lbi) const {
  return lbi2blob_desc_.find(lbi) != lbi2blob_desc_.end();
}

BlobDesc* RegstDesc::MutBlobDesc(const LogicalBlobId& lbi) {
  auto it = lbi2blob_desc_.find(lbi);
  if (it != lbi2blob_desc_.end()) {
    return it->second.get();
  } else {
    return nullptr;
  }
}

const BlobDesc* RegstDesc::SoleBlobDesc() const {
  CHECK_EQ(1, lbi2blob_desc_.size());
  return (*lbi2blob_desc_.begin()).second.get();
}

BlobDesc* RegstDesc::MutSoleBlobDesc() { return const_cast<BlobDesc*>(SoleBlobDesc()); }

void RegstDesc::ForEachLbi(std::function<void(const LogicalBlobId&)> func) const {
  for (const auto& p : lbi2blob_desc_) { func(p.first); }
}

void RegstDesc::ToProto(RegstDescProto* ret) const {
  ret->set_regst_desc_id(regst_desc_id_);
  ret->set_producer_task_id(producer_->task_id());
  for (const TaskNode* consumer : consumers_) { ret->add_consumer_task_id(consumer->task_id()); }
  *(ret->mutable_regst_desc_type()) = regst_desc_type_;
  if (regst_desc_type_.has_data_regst_desc()) {
    DataRegstDesc* data_regst_desc_proto =
        ret->mutable_regst_desc_type()->mutable_data_regst_desc();
    for (const auto& pair : lbi2blob_desc_) {
      LbiBlobDescPair* pb_pair = data_regst_desc_proto->mutable_lbi2blob_desc()->Add();
      *(pb_pair->mutable_lbi()) = pair.first;
      pair.second->ToProto(pb_pair->mutable_blob_desc());
    }
  } else if (regst_desc_type_.has_ctrl_regst_desc()) {
    // do nothing
  } else {
    UNIMPLEMENTED();
  }
  ret->set_register_num(1);
  *(ret->mutable_mem_case()) = mem_case_;
  ret->set_mem_block_id(mem_block_id_);
  ret->set_mem_block_offset(mem_block_offset_);
}

MemoryCase MakeHostMemCase() {
  MemoryCase mem_case;
  mem_case.mutable_host_mem();
  return mem_case;
}

}  // namespace oneflow
