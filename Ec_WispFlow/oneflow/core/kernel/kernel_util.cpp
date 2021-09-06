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
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/memory/memory_case.pb.h"

namespace oneflow {

void AutoMemcpy(void* dst, const void* src, size_t sz,
                const MemoryCase& dst_mem_case, const MemoryCase& src_mem_case) {
  void (*func)(void* dst, const void* src, size_t sz);
  if (src_mem_case.has_host_mem() && dst_mem_case.has_host_mem()) {
    func = &Memcpy<DeviceType::kCPU>;
  } else {
    UNIMPLEMENTED();
  }
  func(dst, src, sz);
}

void AutoMemset(void* dst, const char value, size_t sz,
                const MemoryCase& dst_mem_case) {
  void (*func)(void* dst, const char value, size_t sz);
  if (dst_mem_case.has_host_mem()) {
    func = &Memset<DeviceType::kCPU>;
  } else {
    UNIMPLEMENTED();
  }
  func(dst, value, sz);
}

}  //  namespace oneflow
