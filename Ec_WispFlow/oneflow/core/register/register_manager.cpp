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
#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/memory/memory_allocator.h"

namespace oneflow {

namespace {

void CheckBlobInRegstNotDisabled(const RegstDescProto& regst_desc) {
  CHECK(regst_desc.regst_desc_type().has_data_regst_desc());
}

struct PackedChunkInfo {
  MemoryCase mem_case;
  int64_t size;
  std::vector<const MemBlockProto*> blocks;
  PackedChunkInfo(const MemoryCase& mem) {
    mem_case = mem;
    size = 0;
  }
};

}  // namespace

RegstMgr::RegstMgr(const Plan& plan) {
  HashSet<int64_t> all_block_ids;
  HashMap<int64_t, PackedChunkInfo> zone_id2packed_chunk;
  for (const MemBlockProto& mem_block : plan.block_chunk_list().mem_block()) {
    if (mem_block.mem_size() == 0) { continue; }
    const int64_t mem_block_id = mem_block.mem_block_id();
    CHECK(all_block_ids.insert(mem_block_id).second);
    if (mem_block.has_chunk_id()) {
      UNIMPLEMENTED();
    } else {
      int64_t zone_id = MemoryCaseUtil::GenMemZoneId(mem_block.mem_case());
      if (zone_id2packed_chunk.find(zone_id) == zone_id2packed_chunk.end()) {
        zone_id2packed_chunk.emplace(zone_id, PackedChunkInfo(mem_block.mem_case()));
      }
      PackedChunkInfo* packed_chunk = &(zone_id2packed_chunk.at(zone_id));
      packed_chunk->blocks.push_back(&mem_block);
      packed_chunk->size += mem_block.mem_size();
      CHECK(packed_chunk->mem_case == mem_block.mem_case());
    }
  }

  for (auto& pair : zone_id2packed_chunk) {
    PackedChunkInfo* packed_chunk = &pair.second;
    char* ptr =
        Global<MemoryAllocator>::Get()->Allocate(packed_chunk->mem_case, packed_chunk->size);
    // sort blocks as thrd id
    std::vector<const MemBlockProto*>* blocks = &(packed_chunk->blocks);
    std::sort(blocks->begin(), blocks->end(),
              [](const MemBlockProto* lhs, const MemBlockProto* rhs) {
                if (lhs->thrd_id_hint() == rhs->thrd_id_hint()) {
                  return lhs->mem_block_id() < rhs->mem_block_id();
                }
                return lhs->thrd_id_hint() < rhs->thrd_id_hint();
              });
    int64_t offset = 0;
    for (const MemBlockProto* block : packed_chunk->blocks) {
      CHECK(mem_block_id2ptr_.emplace(block->mem_block_id(), ptr + offset).second);
      offset += block->mem_size();
    }
    CHECK_EQ(offset, packed_chunk->size);
  }

  for (int64_t mem_block_id : all_block_ids) {
    CHECK(mem_block_id2ptr_.find(mem_block_id) != mem_block_id2ptr_.end());
  }

  for (const TaskProto& task : plan.task()) {
    for (const auto& pair : task.produced_regst_desc()) {
      const RegstDescProto& regst_desc = pair.second;
      const int64_t regst_desc_id = regst_desc.regst_desc_id();
      CHECK(regst_desc_id2rt_regst_desc_
                .emplace(regst_desc_id, std::unique_ptr<const RtRegstDesc>(new RtRegstDesc(regst_desc)))
                .second);
    }
  }
}

void RegstMgr::NewRegsts(const RegstDescProto& regst_desc_proto,
                         std::function<void(Regst*)> OneRegstDone) {
  const int64_t regst_desc_id = regst_desc_proto.regst_desc_id();
  const RegstDescTypeProto& regst_desc_type = regst_desc_proto.regst_desc_type();
  const RtRegstDesc* rt_regst_desc = regst_desc_id2rt_regst_desc_.at(regst_desc_id).get();
  char* main_mem_ptr = nullptr;
  int64_t mem_block_id = regst_desc_proto.mem_block_id();
  if (mem_block_id != -1 && mem_block_id2ptr_.find(mem_block_id) != mem_block_id2ptr_.end()) {
    main_mem_ptr = mem_block_id2ptr_.at(mem_block_id) + regst_desc_proto.mem_block_offset();
  }
  std::vector<LbiBlobDescPair> lbi_pairs;
  if (regst_desc_type.has_data_regst_desc()) {
    for (const LbiBlobDescPair& pair : regst_desc_type.data_regst_desc().lbi2blob_desc()) {
      lbi_pairs.push_back(pair);
    }
    std::sort(lbi_pairs.begin(), lbi_pairs.end(), &CompareLbiBlobDescPair);
    CHECK(!lbi_pairs.empty());
  }
  for (int64_t i = 0; i < rt_regst_desc->register_num(); ++i) {
    Regst* regst = new Regst;
    regst->set_regst_desc(rt_regst_desc);
    if (regst_desc_type.has_data_regst_desc()) {
      NewBlobsInOneRegst(lbi_pairs, regst, rt_regst_desc, main_mem_ptr);
      if (main_mem_ptr != nullptr) { main_mem_ptr += rt_regst_desc->MainByteSize4OneRegst(); }
    } else if (regst_desc_type.has_ctrl_regst_desc()) {
      // do nothing
    } else {
      UNIMPLEMENTED();
    }
    OneRegstDone(regst);
  }
}

void RegstMgr::NewBlobsInOneRegst(const std::vector<LbiBlobDescPair>& lbis, Regst* regst,
                                  const RtRegstDesc* rt_regst_desc, char* main_mem_ptr) {
  char* cur_body_pointer = nullptr;
  cur_body_pointer = main_mem_ptr;

  rt_regst_desc->ForEachBlobDescOffsetInOnRegst([&](int64_t ordinal, const LogicalBlobId& lbi,
                                                    const BlobDesc* blob_desc, int64_t body_offset) {
    std::unique_ptr<Blob> blob_ptr;
    blob_ptr.reset(new Blob(regst->regst_desc()->mem_case(), blob_desc,
                            cur_body_pointer + body_offset));
    regst->SetBlobByOrdinal(ordinal, std::move(blob_ptr));
  });
}

const RtRegstDesc& RegstMgr::RegstDesc4RegstDescId(int64_t regst_desc_id) const {
  const auto& it = regst_desc_id2rt_regst_desc_.find(regst_desc_id);
  CHECK(it != regst_desc_id2rt_regst_desc_.end());
  return *it->second;
}

bool RegstMgr::HasRegstDescId(int64_t regst_desc_id) const {
  return regst_desc_id2rt_regst_desc_.find(regst_desc_id) != regst_desc_id2rt_regst_desc_.end();
}

}  // namespace oneflow
