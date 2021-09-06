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
#include "oneflow/core/common/util.h"
#include "oneflow/core/memory/memory_allocator.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

void* MemoryAllocatorImpl::Allocate(const MemoryCase& mem_case, size_t size) {
  void* ptr = nullptr;
  CHECK(mem_case.has_host_mem());
  ptr = aligned_alloc(kHostAlignSize, size);
  CHECK_NOTNULL(ptr);
  return ptr;
}

void MemoryAllocatorImpl::Deallocate(void* ptr, const MemoryCase& mem_case) {
  CHECK(mem_case.has_host_mem());
  free(ptr);
}

void* MemoryAllocatorImpl::AllocateUnPinnedHostMem(size_t size) {
  void* ptr = aligned_alloc(kHostAlignSize, size);
  CHECK_NOTNULL(ptr);
  return ptr;
}

void MemoryAllocatorImpl::DeallocateUnPinnedHostMem(void* ptr) { free(ptr); }

MemoryAllocator::~MemoryAllocator() {
  for (std::function<void()> deleter : deleters_) { deleter(); }
}

char* MemoryAllocator::Allocate(const MemoryCase& mem_case, std::size_t size) {
  const int memset_val = 0;
  char* dptr = static_cast<char*>(MemoryAllocatorImpl::Allocate(mem_case, size));
  CHECK(mem_case.has_host_mem());
  memset(dptr, memset_val, size);

  deleters_.push_front(std::bind(&MemoryAllocator::Deallocate, this, dptr, mem_case));
  return dptr;
}

void MemoryAllocator::Deallocate(char* dptr, const MemoryCase& mem_case) {
  MemoryAllocatorImpl::Deallocate(static_cast<void*>(dptr), mem_case);
}

}  // namespace oneflow
