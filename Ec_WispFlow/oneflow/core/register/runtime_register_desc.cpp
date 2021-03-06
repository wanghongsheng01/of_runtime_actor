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
#include "oneflow/core/register/runtime_register_desc.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

RtRegstDesc::RtRegstDesc(const RegstDescProto& proto) {
  regst_desc_id_ = proto.regst_desc_id();
  producer_actor_id_ = proto.producer_task_id();
  consumers_actor_id_ = PbRf2StdVec(proto.consumer_task_id());
  register_num_ = proto.register_num();
  mem_case_ = proto.mem_case();
  regst_desc_type_ = proto.regst_desc_type();
  if (proto.regst_desc_type().has_data_regst_desc()) {
    const DataRegstDesc& data_regst_desc = proto.regst_desc_type().data_regst_desc();
    std::vector<LbiBlobDescPair> lbi_pairs(
        {data_regst_desc.lbi2blob_desc().cbegin(), data_regst_desc.lbi2blob_desc().cend()});
    std::sort(lbi_pairs.begin(), lbi_pairs.end(), &CompareLbiBlobDescPair);
    CHECK_EQ(lbi_pairs.size(), 1);
    sorted_blob_desc_vec_.reserve(lbi_pairs.size());
    sorted_lbi_vec_.reserve(lbi_pairs.size());
    for (int64_t i = 0; i < lbi_pairs.size(); ++i) {
      const LbiBlobDescPair& pair = lbi_pairs.at(i);
      sorted_blob_desc_vec_.push_back(std::unique_ptr<const BlobDesc>(new BlobDesc(pair.blob_desc())));
      sorted_lbi_vec_.push_back(pair.lbi());
      lbi2blob_desc_ordinal_.emplace(pair.lbi(), i);
    }
  } else {
    UNIMPLEMENTED();
  }
}

int64_t RtRegstDesc::GetOrdinalForLbi(const LogicalBlobId& lbi) const {
  auto it = lbi2blob_desc_ordinal_.find(lbi);
  if (it != lbi2blob_desc_ordinal_.cend()) {
    return it->second;
  } else {
    return -1;
  }
}

const BlobDesc* RtRegstDesc::GetBlobDescFromLbi(const LogicalBlobId& lbi) const {
  auto it = lbi2blob_desc_ordinal_.find(lbi);
  if (it == lbi2blob_desc_ordinal_.end()) {
    return nullptr;
  } else {
    return GetBlobDescByOrdinal(it->second);
  }
}

const BlobDesc* RtRegstDesc::GetBlobDescByOrdinal(int64_t ordinal) const {
  return sorted_blob_desc_vec_.at(ordinal).get();
}

const LogicalBlobId& RtRegstDesc::GetLbiByOrdinal(int64_t ordinal) const {
  return sorted_lbi_vec_.at(ordinal);
}

const BlobDesc* RtRegstDesc::GetSoleBlobDesc() const {
  CHECK_EQ(sorted_blob_desc_vec_.size(), 1);
  return sorted_blob_desc_vec_.at(0).get();
}

size_t RtRegstDesc::TotalByteSize4AllRegst() const {
  return GetSoleBlobDesc()->AlignedTotalByteSize() * register_num_;
}

size_t RtRegstDesc::TotalMainByteSize4AllRegst() const {
  return MainByteSize4OneRegst() * register_num_;
}

size_t RtRegstDesc::MainByteSize4OneRegst() const {
  if (mem_case_.has_device_cuda_mem()) {
    UNIMPLEMENTED();
  } else {
    return GetSoleBlobDesc()->AlignedTotalByteSize();
  }
}

void RtRegstDesc::ForEachBlobDescOffsetInOnRegst(
    const std::function<void(int64_t ordinal, const LogicalBlobId& lbi, const BlobDesc* desc,
                             int64_t body_offset)>& Handler) const {
  int64_t cur_body_offset = 0;
  for (int64_t i = 0; i < sorted_blob_desc_vec_.size(); ++i) {
    const BlobDesc* blob_desc = sorted_blob_desc_vec_.at(i).get();
    const LogicalBlobId& lbi = sorted_lbi_vec_.at(i);
    Handler(i, lbi, blob_desc, cur_body_offset);
    cur_body_offset += blob_desc->AlignedByteSizeOfBlobBody();
  }
}

}  // namespace oneflow
