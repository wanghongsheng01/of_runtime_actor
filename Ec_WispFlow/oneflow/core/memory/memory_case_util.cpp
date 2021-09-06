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
#include "oneflow/core/memory/memory_case_util.h"

namespace oneflow {

bool MemoryCaseUtil::GetCommonMemoryCase(
  const MemoryCase& a, const MemoryCase& b, MemoryCase* common) {
  if (a.has_host_mem() && b.has_host_mem()) {
    *common = a;
    return true;
  } 
  return false;
}

int64_t MemoryCaseUtil::GenMemZoneId(const MemoryCase& mem_case) {
  // [0, 127] = GPU device mem
  // [128] = CPU host mem
  // [129, 256] = CPU host mem used by CUDA with device id
  // [257, ...] Other Device
  if (mem_case.has_host_mem()) {
    return 128;  // CPU host mem
  }
  UNIMPLEMENTED();
  return -1;
}

int64_t MemoryCaseUtil::GenMemZoneUniqueId(int64_t machine_id, const MemoryCase& mem_case) {
  return (machine_id << 32) | (MemoryCaseUtil::GenMemZoneId(mem_case));
}

std::shared_ptr<MemoryCase> MemoryCaseUtil::MakeMemCase(const DeviceType device_type,
                                                        const int64_t device_id) {
  const auto mem_case = std::make_shared<MemoryCase>();
  if (device_type == DeviceType::kCPU) {
    mem_case->mutable_host_mem();
  } else {
    UNIMPLEMENTED();
  }
  return mem_case;
}

}  // namespace oneflow
